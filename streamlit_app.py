import json
import math
import os
import io
import zipfile
import tempfile
import re
import shutil
import tempfile
import traceback
import uuid
import hashlib
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path

import cv2
import imageio_ffmpeg
import numpy as np
import pandas as pd
import requests
import streamlit as st
import toml
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.signal import find_peaks


def _safe_filename(value):
    s = str(value or "").strip()
    if not s:
        return "sin_nombre"
    s = re.sub(r'[<>:"/\\|?*]+', "_", s)
    s = re.sub(r"\s+", "_", s)
    return s[:80]


def _metrics_dataframe(metrics):
    rows = []
    for m in metrics or []:
        rows.append({
            "key": m.get("key"),
            "label": m.get("label"),
            "value": m.get("value"),
            "unit": m.get("unit"),
            "quality": m.get("quality"),
            "notes": m.get("notes"),
        })
    return pd.DataFrame(rows)


def _plot_column_png(chart_df, column, title=None):
    """
    Genera un PNG en memoria. V74 preserva los huecos de cambio de sentido:
    los NaN no se comprimen y la zona excluida se sombrea, por lo que nunca se
    dibuja una línea artificial entre el tramo posterior y el anterior.
    """
    if chart_df is None or column not in chart_df.columns:
        return None
    y = pd.to_numeric(chart_df[column], errors="coerce").to_numpy(float)
    if "time_s" in chart_df.columns:
        x = pd.to_numeric(chart_df["time_s"], errors="coerce").to_numpy(float)
        xlabel = "Tiempo (s)"
    else:
        x = np.arange(len(chart_df), dtype=float)
        xlabel = "Muestra"

    valid = np.isfinite(x) & np.isfinite(y)
    if int(valid.sum()) < 2:
        return None

    # Mantener NaN en Y para que Matplotlib corte físicamente la curva.
    y_plot = y.copy()
    y_plot[~np.isfinite(x)] = np.nan

    fig, ax = plt.subplots(figsize=(10, 4.8))
    ax.plot(x, y_plot, linewidth=1.6)

    # V74 · visualización explícita del dominio excluido.
    if "straight_walking_valid" in chart_df.columns:
        sm = chart_df["straight_walking_valid"].fillna(False).astype(bool).to_numpy()
        if len(sm) == len(x):
            in_run = False; a = 0
            for i, ok in enumerate(sm):
                if (not ok) and not in_run:
                    a = i; in_run = True
                end_run = in_run and (ok or i == len(sm)-1)
                if end_run:
                    b = i if ok else i + 1
                    xa = x[a] if a < len(x) and np.isfinite(x[a]) else np.nan
                    xb_idx = min(max(b-1, a), len(x)-1)
                    xb = x[xb_idx] if np.isfinite(x[xb_idx]) else np.nan
                    if np.isfinite(xa) and np.isfinite(xb):
                        ax.axvspan(xa, xb, alpha=0.10, hatch="//", label="Cambio de sentido / transición excluida")
                    in_run = False
            # Evitar leyendas repetidas si hay más de una zona excluida.
            handles, labels = ax.get_legend_handles_labels()
            if labels:
                uniq = {}
                for h,l in zip(handles,labels): uniq.setdefault(l,h)
                ax.legend(uniq.values(), uniq.keys(), loc="best", fontsize=8)

    ax.set_title(title or str(column))
    ax.set_xlabel(xlabel)
    ax.set_ylabel(str(column))
    ax.grid(True, alpha=0.25)
    fig.tight_layout()

    bio = io.BytesIO()
    fig.savefig(bio, format="png", dpi=160, bbox_inches="tight")
    plt.close(fig)
    bio.seek(0)
    return bio.getvalue()


def _add_chart_bundle_to_zip(zf, chart_df, folder_name):
    if chart_df is None or not isinstance(chart_df, pd.DataFrame) or chart_df.empty:
        return 0

    # Datos fuente del gráfico
    zf.writestr(
        f"{folder_name}/datos_graficos.csv",
        chart_df.to_csv(index=False).encode("utf-8-sig")
    )

    count = 0
    ignore = {"frame", "time_s", "straight_walking_valid", "straight_block_id"}
    for col in chart_df.columns:
        if col in ignore:
            continue
        try:
            png = _plot_column_png(chart_df, col, title=col)
            if png:
                zf.writestr(
                    f"{folder_name}/graficos/{count+1:02d}_{_safe_filename(col)}.png",
                    png
                )
                count += 1
        except Exception:
            continue
    return count


# === V58 CORREGIDA · persistencia/exportación global V56 + V58 ===
def _v58c_jsonable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, pd.DataFrame):
        return value.to_dict(orient="records")
    if isinstance(value, dict):
        return {str(k): _v58c_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_v58c_jsonable(v) for v in value]
    return value


def _v58c_add_bytes_to_zip(zip_obj, arcname, payload):
    if payload is None:
        return False
    if isinstance(payload, str):
        payload = payload.encode("utf-8")
    zip_obj.writestr(arcname, payload)
    return True


def _v58c_export_v56_v58_to_zip(zip_obj):
    """
    Añade al ZIP global la triangulación V56 persistida y la calibración V58.
    No recalcula V56 ni modifica Tsai/DLT/QC.
    """
    exported = {"v56": False, "v58": False}

    # V56: priorizar el payload congelado de V56 CORREGIDA 3.
    v56_payload = st.session_state.get("v56_export_payload")
    if isinstance(v56_payload, dict) and v56_payload:
        # Si el payload ya contiene archivos serializados, conservarlos.
        file_map = v56_payload.get("files")
        if isinstance(file_map, dict):
            for name, data in file_map.items():
                safe = str(name).replace("\\", "/").split("/")[-1]
                _v58c_add_bytes_to_zip(zip_obj, f"08_triangulacion_v56/{safe}", data)
                exported["v56"] = True

        # También guardar el payload completo como respaldo auditable.
        try:
            zip_obj.writestr(
                "08_triangulacion_v56/payload_v56_persistido.json",
                json.dumps(_v58c_jsonable(v56_payload), ensure_ascii=False, indent=2)
            )
            exported["v56"] = True
        except Exception:
            pass

    # Fallback V56 live solo si no hubo payload persistido.
    if not exported["v56"]:
        v56_live = st.session_state.get("v56_triangulation")
        if isinstance(v56_live, dict) and v56_live:
            try:
                zip_obj.writestr(
                    "08_triangulacion_v56/triangulacion_v56_estado.json",
                    json.dumps(_v58c_jsonable(v56_live), ensure_ascii=False, indent=2)
                )
                exported["v56"] = True
            except Exception:
                pass

    # V58: exportar calibración calculada o perfil activo recargado.
    v58_out = st.session_state.get("v58_calibration")
    active = st.session_state.get("v58_active_profile")

    profile = None
    detection_df = None
    summary = None
    if isinstance(v58_out, dict):
        profile = v58_out.get("profile")
        detection_df = v58_out.get("detection_df")
        summary = v58_out.get("summary")
    if not isinstance(profile, dict) and isinstance(active, dict):
        profile = active

    if isinstance(profile, dict):
        zip_obj.writestr(
            "09_calibracion_estereo_v58/perfil_calibracion_v58.json",
            json.dumps(_v58c_jsonable(profile), ensure_ascii=False, indent=2)
        )
        exported["v58"] = True

    if isinstance(summary, dict):
        zip_obj.writestr(
            "09_calibracion_estereo_v58/resumen_calibracion_v58.json",
            json.dumps(_v58c_jsonable(summary), ensure_ascii=False, indent=2)
        )
        exported["v58"] = True

    if isinstance(detection_df, pd.DataFrame) and not detection_df.empty:
        zip_obj.writestr(
            "09_calibracion_estereo_v58/qc_detecciones_charuco_v58.csv",
            detection_df.to_csv(index=False).encode("utf-8-sig")
        )
        exported["v58"] = True

    # Manifiesto explícito incluso si todavía no existe una calibración V58.
    manifest = {
        "version": "58-corregida",
        "v56_exported": bool(exported["v56"]),
        "v58_exported": bool(exported["v58"]),
        "v56_source_priority": "v56_export_payload -> v56_triangulation",
        "v58_source_priority": "v58_calibration -> v58_active_profile",
        "note": "V56 no se recalcula. V55/V56 Tsai y DLT permanecen congeladas."
    }
    zip_obj.writestr(
        "09_calibracion_estereo_v58/manifiesto_exportacion_v58.json",
        json.dumps(manifest, ensure_ascii=False, indent=2)
    )
    return exported

def build_export_zip(
    metrics,
    chart=None,
    chart_front=None,
    chart_lateral=None,
    technical_report="",
    patient_report="",
    annotated_video_bytes=None,
    annotated_video2_bytes=None,
    patient_code="",
    record_name="",
    patient_age=0,
    patient_sex="No especificado",
    record_date=None,
    app_version="",
):
    """
    Crea un paquete ZIP totalmente en memoria.
    No persiste gráficos, vídeos ni resultados pesados en Supabase.
    """
    out = io.BytesIO()

    meta = {
        "app_version": app_version,
        "patient_code": patient_code,
        "record_name": record_name,
        "patient_age": int(patient_age or 0),
        "patient_sex": patient_sex,
        "record_date": record_date.isoformat() if hasattr(record_date, "isoformat") else str(record_date or ""),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    }

    with zipfile.ZipFile(out, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(
            "00_metadatos.json",
            json.dumps(meta, ensure_ascii=False, indent=2).encode("utf-8")
        )

        mdf = _metrics_dataframe(metrics)
        if not mdf.empty:
            zf.writestr(
                "01_resultados/resultados_metricas.csv",
                mdf.to_csv(index=False).encode("utf-8-sig")
            )
            zf.writestr(
                "01_resultados/resultados_metricas.json",
                json.dumps(mdf.to_dict(orient="records"), ensure_ascii=False, indent=2, default=str).encode("utf-8")
            )
            try:
                annex_df = build_annex_dataframe(metrics)
                if not annex_df.empty:
                    zf.writestr("01_resultados/anexo_diccionario_variables.csv", annex_df.to_csv(index=False).encode("utf-8-sig"))
            except Exception:
                pass

        if technical_report:
            zf.writestr("02_informes/informe_tecnico.txt", technical_report.encode("utf-8"))
        if patient_report:
            zf.writestr("02_informes/informe_paciente.txt", patient_report.encode("utf-8"))

        # V42 · auditoría temporal exhaustiva incluida en el paquete exportado.
        try:
            _a42 = st.session_state.get("biplanar_v42_triplet_audit") or []
            _c42 = st.session_state.get("biplanar_v42_candidate_audit") or []
            if _a42:
                zf.writestr("01_resultados/v42_auditoria_tripletes.csv",
                            pd.DataFrame(_a42).to_csv(index=False).encode("utf-8-sig"))
            if _c42:
                zf.writestr("01_resultados/v42_auditoria_candidatos_periodicos.csv",
                            pd.DataFrame(_c42).to_csv(index=False).encode("utf-8-sig"))
            _v43 = st.session_state.get("biplanar_v43_closure_cycles") or []
            if _v43:
                zf.writestr("01_resultados/v43_ciclos_homologos_por_cierre.csv",
                            pd.DataFrame(_v43).to_csv(index=False).encode("utf-8-sig"))
            _v44 = st.session_state.get("biplanar_v44_chains") or []
            _v44a = st.session_state.get("biplanar_v44_chain_audit") or []
            if _v44:
                zf.writestr("01_resultados/v44_cadenas_ciclos_consecutivos.csv",
                            pd.DataFrame(_v44).to_csv(index=False).encode("utf-8-sig"))
            if _v44a:
                zf.writestr("01_resultados/v44_auditoria_cadenas.csv",
                            pd.DataFrame(_v44a).to_csv(index=False).encode("utf-8-sig"))
            _v45near = st.session_state.get("biplanar_v45_near_prediction_candidates") or []
            _v45_summary = {
                "status": str(st.session_state.get("biplanar_v45_status","")),
                "reason": str(st.session_state.get("biplanar_v45_reason","")),
            }
            zf.writestr("01_resultados/v45_evaluabilidad_segundo_ciclo.csv",
                        pd.DataFrame([_v45_summary]).to_csv(index=False).encode("utf-8-sig"))
            if _v45near:
                zf.writestr("01_resultados/v45_candidatos_proximos_ic_esperado.csv",
                            pd.DataFrame(_v45near).to_csv(index=False).encode("utf-8-sig"))
        except Exception:
            pass

        # V48 · perfil biomecánico biplanar sobre un ciclo físico homólogo.
        try:
            _v48 = st.session_state.get("biplanar_v48_profile")
            if isinstance(_v48, dict):
                _v48s = _v48.get("summary") or {}
                _v48c = _v48.get("curves")
                if _v48s:
                    zf.writestr("06_biplanar_v48/resumen_ciclo_homologo.json",
                                json.dumps(_v48s, ensure_ascii=False, indent=2, default=str).encode("utf-8"))
                    zf.writestr("06_biplanar_v48/resumen_ciclo_homologo.csv",
                                pd.DataFrame([_v48s]).to_csv(index=False).encode("utf-8-sig"))
                if isinstance(_v48c, pd.DataFrame) and not _v48c.empty:
                    zf.writestr("06_biplanar_v48/curvas_ciclo_biplanar_normalizado.csv",
                                _v48c.to_csv(index=False).encode("utf-8-sig"))
                # V50 · copia canónica del perfil + auditoría multiseñal del TO.
                if _v48s:
                    zf.writestr("06_biplanar_v50/resumen_to_biplanar_v50.json",
                                json.dumps(_v48s, ensure_ascii=False, indent=2, default=str).encode("utf-8"))
                    zf.writestr("06_biplanar_v50/resumen_to_biplanar_v50.csv",
                                pd.DataFrame([_v48s]).to_csv(index=False).encode("utf-8-sig"))
                _v50a=_v48.get("v50_to_audit")
                if isinstance(_v50a,pd.DataFrame) and not _v50a.empty:
                    zf.writestr("06_biplanar_v50/auditoria_multisenal_to.csv",
                                _v50a.to_csv(index=False).encode("utf-8-sig"))
                # V51 · salida canónica del detector TO jerárquico dependiente de perspectiva.
                if _v48s:
                    zf.writestr("06_biplanar_v51/resumen_to_jerarquico_v51.json",
                                json.dumps(_v48s, ensure_ascii=False, indent=2, default=str).encode("utf-8"))
                    zf.writestr("06_biplanar_v51/resumen_to_jerarquico_v51.csv",
                                pd.DataFrame([_v48s]).to_csv(index=False).encode("utf-8-sig"))
                _v51a=_v48.get("v51_to_audit")
                if isinstance(_v51a,pd.DataFrame) and not _v51a.empty:
                    zf.writestr("06_biplanar_v51/auditoria_to_jerarquico.csv",
                                _v51a.to_csv(index=False).encode("utf-8-sig"))
                # V52 · ciclo clínico completo IC→TO→IC y métricas por fase.
                if _v48s:
                    _v52s={k:v for k,v in _v48s.items() if str(k).startswith("v52_")}
                    if _v52s:
                        zf.writestr("06_biplanar_v52/resumen_ciclo_clinico_v52.json",
                                    json.dumps(_v52s, ensure_ascii=False, indent=2, default=str).encode("utf-8"))
                        zf.writestr("06_biplanar_v52/resumen_ciclo_clinico_v52.csv",
                                    pd.DataFrame([_v52s]).to_csv(index=False).encode("utf-8-sig"))
                _v52c=_v48.get("curves")
                if isinstance(_v52c,pd.DataFrame) and not _v52c.empty:
                    zf.writestr("06_biplanar_v52/curvas_ciclo_clinico_0_100.csv",
                                _v52c.to_csv(index=False).encode("utf-8-sig"))
                _v52m=_v48.get("v52_phase_metrics")
                if isinstance(_v52m,pd.DataFrame) and not _v52m.empty:
                    zf.writestr("06_biplanar_v52/resumen_variables_por_fase.csv",
                                _v52m.to_csv(index=False).encode("utf-8-sig"))
                _v52e=_v48.get("v52_events")
                if isinstance(_v52e,pd.DataFrame) and not _v52e.empty:
                    zf.writestr("06_biplanar_v52/eventos_ciclo_clinico.csv",
                                _v52e.to_csv(index=False).encode("utf-8-sig"))

                # V53 CORREGIDA · interpretación biomecánica biplanar exportable.
                # Se construye aquí directamente desde los resultados V52/V51 ya calculados,
                # para que la exportación no dependa del orden de renderizado de Streamlit.
                _v53_summary = {}
                if _v48s:
                    _v53_summary = {
                        "version": "53",
                        "scope": "Interpretación biomecánica biplanar 2D no diagnóstica",
                        "cycle_duration_s": _v48s.get("v52_cycle_duration_s"),
                        "stance_pct": _v48s.get("v52_stance_pct"),
                        "swing_pct": _v48s.get("v52_swing_pct"),
                        "to_pct": (
                            _v48s.get("v52_to_pct")
                            if _v48s.get("v52_to_pct") is not None
                            else _v48s.get("v51_selected_to_pct")
                        ),
                        "to_status": (
                            _v48s.get("v51_status")
                            or _v48s.get("v51_to_status")
                            or _v48s.get("v52_to_status")
                        ),
                        "quality": (
                            _v48s.get("v52_quality")
                            or _v48s.get("v51_quality")
                            or _v48s.get("quality")
                        ),
                        "interpretation_rule": (
                            "Describe el patrón observado. No clasifica automáticamente "
                            "normalidad, patología, diagnóstico ni pronóstico."
                        ),
                    }

                _v53_obs = []
                if isinstance(_v52m, pd.DataFrame) and not _v52m.empty:
                    # Exportamos primero la evidencia cuantitativa completa.
                    zf.writestr(
                        "06_biplanar_v53/medicion_observada_variables_v53.csv",
                        _v52m.to_csv(index=False).encode("utf-8-sig")
                    )

                    # Generamos texto cuantitativo explicable por variable, sin umbrales normativos.
                    _cols = {str(c).lower(): c for c in _v52m.columns}

                    def _v53_col(*names):
                        for _n in names:
                            if _n.lower() in _cols:
                                return _cols[_n.lower()]
                        return None

                    _var_col = _v53_col("variable", "name", "metrica")
                    _phase_col = _v53_col("phase", "fase")
                    _rom_col = _v53_col("rom_deg", "rom", "rango_movimiento_deg")
                    _mean_col = _v53_col("mean_deg", "mean", "media_deg")
                    _min_col = _v53_col("min_deg", "min", "minimo_deg")
                    _max_col = _v53_col("max_deg", "max", "maximo_deg")

                    if _var_col is not None:
                        for _var, _g in _v52m.groupby(_var_col, dropna=False):
                            _entry = {
                                "variable": str(_var),
                                "medicion_observada": [],
                                "interpretacion": (
                                    "Patrón cuantitativo observado en el mismo ciclo físico 0–100 %. "
                                    "Debe interpretarse clínicamente junto con la perspectiva 2D, "
                                    "la calidad de tracking y la incertidumbre de eventos."
                                )
                            }
                            for _, _r in _g.iterrows():
                                _phase = str(_r.get(_phase_col, "")) if _phase_col is not None else ""
                                _parts = []
                                if _mean_col is not None and pd.notna(_r.get(_mean_col)):
                                    _parts.append(f"media={float(_r.get(_mean_col)):.2f}°")
                                if _min_col is not None and pd.notna(_r.get(_min_col)):
                                    _parts.append(f"mín={float(_r.get(_min_col)):.2f}°")
                                if _max_col is not None and pd.notna(_r.get(_max_col)):
                                    _parts.append(f"máx={float(_r.get(_max_col)):.2f}°")
                                if _rom_col is not None and pd.notna(_r.get(_rom_col)):
                                    _parts.append(f"ROM={float(_r.get(_rom_col)):.2f}°")
                                if _parts:
                                    _entry["medicion_observada"].append({
                                        "fase": _phase,
                                        "valores": "; ".join(_parts)
                                    })
                            _v53_obs.append(_entry)

                # Eventos V52 también forman parte de la evidencia observada de V53.
                if isinstance(_v52e, pd.DataFrame) and not _v52e.empty:
                    zf.writestr(
                        "06_biplanar_v53/eventos_referencia_v53.csv",
                        _v52e.to_csv(index=False).encode("utf-8-sig")
                    )

                if _v53_summary or _v53_obs:
                    _v53_payload = {
                        "resumen": _v53_summary,
                        "variables": _v53_obs,
                        "integracion_biomecanica": (
                            "La vista frontal/posterior aporta información del plano frontal y "
                            "la vista lateral aporta información sagital, ambas referidas al mismo "
                            "ciclo físico. Esta integración sigue siendo 2D multiplanar y no una "
                            "reconstrucción 3D."
                        ),
                        "limitaciones": [
                            "No establece normalidad o patología de forma automática.",
                            "La geometría 2D depende de la perspectiva y colocación de las cámaras.",
                            "La calidad del TO se hereda del detector jerárquico V51.",
                            "No equivale a plataforma de fuerzas, dinámica inversa ni reconstrucción 3D."
                        ],
                    }
                    zf.writestr(
                        "06_biplanar_v53/resumen_interpretacion_v53.json",
                        json.dumps(_v53_payload, ensure_ascii=False, indent=2, default=str).encode("utf-8")
                    )

                    # Resumen legible en TXT.
                    _lines = [
                        "PhysioSentinel Gait · V53",
                        "INTERPRETACIÓN BIOMECÁNICA BIPLANAR DEL CICLO",
                        "",
                        "MEDICIÓN OBSERVADA",
                    ]
                    for _item in _v53_obs:
                        _lines.append(f"- {_item['variable']}")
                        for _x in _item["medicion_observada"]:
                            _lines.append(f"  · {_x['fase']}: {_x['valores']}")
                    _lines += [
                        "",
                        "INTERPRETACIÓN",
                        "Las variables frontales/posteriores y sagitales se relacionan temporalmente "
                        "sobre el mismo ciclo físico 0–100 %. La lectura es descriptiva y explicable; "
                        "no clasifica automáticamente normalidad, patología, diagnóstico ni pronóstico.",
                        "",
                        "ALCANCE",
                        "Análisis 2D (Bidimensional) multiplanar. No equivale a reconstrucción 3D "
                        "(Tridimensional), plataforma de fuerzas ni dinámica inversa.",
                    ]
                    zf.writestr(
                        "06_biplanar_v53/informe_interpretacion_v53.txt",
                        "\n".join(_lines).encode("utf-8")
                    )

                # V54 CORREGIDA 3 · preparación geométrica pre-3D.
                # Se deriva SIEMPRE desde el perfil V48 vivo para evitar estados
                # antiguos creados antes del cálculo.
                _v54_v48_live = st.session_state.get("biplanar_v48_profile")
                _v54_sum_live = (
                    _v54_v48_live.get("summary") or {}
                    if isinstance(_v54_v48_live, dict) else {}
                )
                _v54_cycle_live = bool(
                    isinstance(_v54_sum_live, dict)
                    and _v54_sum_live.get("selected_pair_index") is not None
                    and str(_v54_sum_live.get("front_side", "")).strip()
                    and str(_v54_sum_live.get("lateral_side", "")).strip()
                )
                _v54_payload = {
                    "version": "54-corregida-3",
                    "purpose": "Preparación geométrica pre-3D",
                    "sync_layer": "V47 congelada",
                    "homologous_cycle_available": _v54_cycle_live,
                    "homologous_pair_index": _v54_sum_live.get("selected_pair_index") if _v54_cycle_live else None,
                    "front_side": _v54_sum_live.get("front_side") if _v54_cycle_live else None,
                    "lateral_side": _v54_sum_live.get("lateral_side") if _v54_cycle_live else None,
                    "integration_confidence": _v54_sum_live.get("integration_confidence") if _v54_cycle_live else None,
                    "ic_alignment_delta_ms": _v54_sum_live.get("ic_alignment_delta_ms") if _v54_cycle_live else None,
                    "front_cycle_s": _v54_sum_live.get("front_cycle_s") if _v54_cycle_live else None,
                    "lateral_cycle_s": _v54_sum_live.get("lateral_cycle_s") if _v54_cycle_live else None,
                    "variables_front": _v54_sum_live.get("variables_front") if _v54_cycle_live else None,
                    "variables_lateral": _v54_sum_live.get("variables_lateral") if _v54_cycle_live else None,
                    "intrinsics_cam01": "pendiente",
                    "intrinsics_cam02": "pendiente",
                    "extrinsics_cam01_cam02": "pendiente",
                    "simultaneous_fov_overlap": "por verificar",
                    "triangulation_3d_enabled": False,
                    "source": "biplanar_v48_profile leído en el momento de exportar",
                }
                # Mantener session_state consistente con lo exportado.
                st.session_state.v54_geometry_readiness = _v54_payload

                zf.writestr(
                    "06_biplanar_v54/preparacion_geometrica_v54.json",
                    json.dumps(_v54_payload, ensure_ascii=False, indent=2, default=str).encode("utf-8")
                )
                zf.writestr(
                    "06_biplanar_v54/preparacion_geometrica_v54.csv",
                    pd.DataFrame([_v54_payload]).to_csv(index=False).encode("utf-8-sig")
                )

                # V55 · calibración geométrica de cámaras.
                _v55 = st.session_state.get("v55_calibration")
                if isinstance(_v55, dict):
                    _v55_summary = dict(_v55.get("summary") or {})

                    # V55 CORREGIDA: actualizar el estado temporal EN EL MOMENTO DE EXPORTAR.
                    # Evita exportar un false antiguo si V48 se calculó después de cargar los XML.
                    _v48_export = st.session_state.get("biplanar_v48_profile") or {}
                    _v48_export_summary = (
                        (_v48_export.get("summary") or {})
                        if isinstance(_v48_export, dict) else {}
                    )
                    _cycle_export_ok = bool(
                        _v48_export_summary.get("selected_pair_index") is not None
                        and _v48_export_summary.get("front_side")
                        and _v48_export_summary.get("lateral_side")
                    )

                    if _cycle_export_ok:
                        _v55_summary["homologous_cycle_available"] = True
                        _v55_summary["homologous_pair_index"] = _v48_export_summary.get("selected_pair_index")
                        _v55_summary["front_side"] = _v48_export_summary.get("front_side")
                        _v55_summary["lateral_side"] = _v48_export_summary.get("lateral_side")
                        _v55_summary["integration_confidence"] = _v48_export_summary.get("integration_confidence")
                        _v55_summary["ic_alignment_delta_ms"] = _v48_export_summary.get("ic_alignment_delta_ms")
                        _v55_summary["front_cycle_s"] = _v48_export_summary.get("front_cycle_s")
                        _v55_summary["lateral_cycle_s"] = _v48_export_summary.get("lateral_cycle_s")
                        _v55_summary["cycle_source"] = "biplanar_v48_profile leído en el momento de exportar"

                    _calib_ready_export = bool(
                        _v55_summary.get("cam01_valid")
                        and _v55_summary.get("cam02_valid")
                        and _v55_summary.get("relative_geometry_valid")
                    )
                    _v55_summary["projection_matrices_ready"] = bool(
                        _v55_summary.get("cam01_valid") and _v55_summary.get("cam02_valid")
                    )
                    _v55_summary["v56_triangulation_ready"] = bool(
                        _cycle_export_ok and _calib_ready_export
                    )
                    _v55_summary["triangulation_3d_enabled"] = False
                    _v55_summary["version"] = "55-corregida"

                    if _v55_summary:
                        zf.writestr(
                            "07_calibracion_v55/resumen_calibracion_v55.json",
                            json.dumps(_v55_summary, ensure_ascii=False, indent=2, default=str).encode("utf-8")
                        )
                        zf.writestr(
                            "07_calibracion_v55/resumen_calibracion_v55.csv",
                            pd.DataFrame([_v55_summary]).to_csv(index=False).encode("utf-8-sig")
                        )

                    for _cam_key, _folder in [("cam01", "cam01"), ("cam02", "cam02")]:
                        _cam = _v55.get(_cam_key)
                        if isinstance(_cam, dict):
                            _params = _cam.get("parameters") or {}
                            if _params:
                                zf.writestr(
                                    f"07_calibracion_v55/{_folder}/parametros_tsai.csv",
                                    pd.DataFrame(
                                        [{"parametro": k, "valor": v} for k, v in _params.items()]
                                    ).to_csv(index=False).encode("utf-8-sig")
                                )
                            for _name in ["K", "R", "t", "P", "camera_center_world"]:
                                _mat = _cam.get(_name)
                                if _mat is not None:
                                    _arr = np.asarray(_mat, dtype=float)
                                    if _arr.ndim == 1:
                                        _arr = _arr.reshape(1, -1)
                                    zf.writestr(
                                        f"07_calibracion_v55/{_folder}/{_name}.csv",
                                        pd.DataFrame(_arr).to_csv(index=False, header=False).encode("utf-8-sig")
                                    )

                    _rel = _v55.get("relative")
                    if isinstance(_rel, dict):
                        for _name in ["R_cam01_to_cam02", "t_cam01_to_cam02"]:
                            _mat = _rel.get(_name)
                            if _mat is not None:
                                _arr = np.asarray(_mat, dtype=float)
                                if _arr.ndim == 1:
                                    _arr = _arr.reshape(1, -1)
                                zf.writestr(
                                    f"07_calibracion_v55/relativa/{_name}.csv",
                                    pd.DataFrame(_arr).to_csv(index=False, header=False).encode("utf-8-sig")
                                )
                        zf.writestr(
                            "07_calibracion_v55/relativa/resumen_relativo.json",
                            json.dumps(_rel, ensure_ascii=False, indent=2, default=str).encode("utf-8")
                        )


                # V56 CORREGIDA 3 · triangulación 3D HALPE26.
                # El payload serializado se genera EN EL MOMENTO de pulsar
                # "Triangular" y por tanto representa la ejecución más reciente.
                # Para evitar exportar una estructura viva obsoleta de un rerun
                # anterior, el ZIP global prioriza SIEMPRE ese payload.
                _v56_payload = st.session_state.get("v56_export_payload") or {}
                _v56 = st.session_state.get("v56_triangulation")

                if isinstance(_v56_payload, dict) and _v56_payload:
                    if _v56_payload.get("summary_json"):
                        zf.writestr(
                            "08_triangulacion_v56/resumen_triangulacion_v56.json",
                            _v56_payload["summary_json"].encode("utf-8")
                        )
                    if _v56_payload.get("summary_csv"):
                        zf.writestr(
                            "08_triangulacion_v56/resumen_triangulacion_v56.csv",
                            _v56_payload["summary_csv"].encode("utf-8-sig")
                        )
                    if _v56_payload.get("points3d_csv"):
                        zf.writestr(
                            "08_triangulacion_v56/puntos_3d_halpe26_v56.csv",
                            _v56_payload["points3d_csv"].encode("utf-8-sig")
                        )
                    if _v56_payload.get("frame_qc_csv"):
                        zf.writestr(
                            "08_triangulacion_v56/control_calidad_por_frame_v56.csv",
                            _v56_payload["frame_qc_csv"].encode("utf-8-sig")
                        )
                    if _v56_payload.get("landmark_qc_csv"):
                        zf.writestr(
                            "08_triangulacion_v56/control_calidad_por_landmark_v56.csv",
                            _v56_payload["landmark_qc_csv"].encode("utf-8-sig")
                        )

                elif isinstance(_v56, dict):
                    # Fallback únicamente si todavía no existe payload serializado.
                    _v56_summary = dict(_v56.get("summary") or {})
                    if _v56_summary:
                        zf.writestr(
                            "08_triangulacion_v56/resumen_triangulacion_v56.json",
                            json.dumps(_v56_summary, ensure_ascii=False, indent=2, default=str).encode("utf-8")
                        )
                        zf.writestr(
                            "08_triangulacion_v56/resumen_triangulacion_v56.csv",
                            pd.DataFrame([_v56_summary]).to_csv(index=False).encode("utf-8-sig")
                        )

                    _pts3d = _v56.get("points3d")
                    if isinstance(_pts3d, pd.DataFrame) and not _pts3d.empty:
                        zf.writestr(
                            "08_triangulacion_v56/puntos_3d_halpe26_v56.csv",
                            _pts3d.to_csv(index=False).encode("utf-8-sig")
                        )

                    _frame_qc = _v56.get("frame_qc")
                    if isinstance(_frame_qc, pd.DataFrame) and not _frame_qc.empty:
                        zf.writestr(
                            "08_triangulacion_v56/control_calidad_por_frame_v56.csv",
                            _frame_qc.to_csv(index=False).encode("utf-8-sig")
                        )

                    _landmark_qc = _v56.get("landmark_qc")
                    if isinstance(_landmark_qc, pd.DataFrame) and not _landmark_qc.empty:
                        zf.writestr(
                            "08_triangulacion_v56/control_calidad_por_landmark_v56.csv",
                            _landmark_qc.to_csv(index=False).encode("utf-8-sig")
                        )

        except Exception:
            pass

        graph_count = 0
        graph_count += _add_chart_bundle_to_zip(zf, chart, "03_graficos/vista_unica")
        graph_count += _add_chart_bundle_to_zip(zf, chart_front, "03_graficos/frontal")
        graph_count += _add_chart_bundle_to_zip(zf, chart_lateral, "03_graficos/lateral")

        if annotated_video_bytes:
            zf.writestr("04_video_anotado/video_anotado_cam01.mp4", annotated_video_bytes)
        if annotated_video2_bytes:
            zf.writestr("04_video_anotado/video_anotado_cam02.mp4", annotated_video2_bytes)

        # v0.10.0 · exportación del analizador del ciclo si fue generado
        try:
            if st.session_state.get("cycle_kinematic_png"):
                zf.writestr("05_ciclo_marcha/curva_cinematica_normalizada.png", st.session_state["cycle_kinematic_png"])
            if st.session_state.get("cycle_phase_png"):
                zf.writestr("05_ciclo_marcha/fases_por_extremidad.png", st.session_state["cycle_phase_png"])
            phase_table = st.session_state.get("cycle_phase_table")
            if isinstance(phase_table, pd.DataFrame) and not phase_table.empty:
                zf.writestr("05_ciclo_marcha/resumen_fases.csv", phase_table.to_csv(index=False).encode("utf-8-sig"))
        except Exception:
            pass

        manifest = (
            f"PhysioSentinel Gait {app_version}\n"
            f"Paciente/código: {patient_code}\n"
            f"Registro: {record_name}\n"
            f"Gráficos PNG exportados: {graph_count}\n\n"
            "Este paquete se genera bajo demanda en memoria y no implica "
            "almacenamiento permanente de gráficos o vídeos en Supabase.\n"
        )
        _v58c_export_v56_v58_to_zip(zf)
        zf.writestr("LEEME_EXPORTACION.txt", manifest.encode("utf-8"))

    out.seek(0)
    return out.getvalue()





def _valid_dataframe(obj):
    return isinstance(obj, pd.DataFrame) and not obj.empty



def _support_summary_positions_to_video_frames(summary, seg):
    """
    v0.10.4 · _support_cycle_summary trabaja sobre posiciones 0..N-1 de la
    máscara. Para sincronizar con vídeo y curvas hay que convertir IC/TO/nextIC
    a los números de frame reales contenidos en seg["frame"].
    """
    if not isinstance(summary, dict) or not _valid_dataframe(seg):
        return summary
    frame_values = pd.to_numeric(seg["frame"], errors="coerce").to_numpy()
    n = len(frame_values)
    out = dict(summary)
    for key in ("ic", "to", "next_ic", "starts", "ends"):
        arr = np.asarray(summary.get(key, []), dtype=int)
        mapped = []
        for p in arr:
            if n == 0:
                continue
            # `ends` puede ser exclusivo == n; mapearlo al último frame + 1
            if key == "ends" and p >= n:
                if np.isfinite(frame_values[-1]):
                    mapped.append(int(frame_values[-1]) + 1)
                continue
            q = int(np.clip(p, 0, n-1))
            if np.isfinite(frame_values[q]):
                mapped.append(int(frame_values[q]))
        out[key] = np.asarray(mapped, dtype=int)
    return out


def _pair_cycles(left_cycles, right_cycles):
    """
    Empareja cada ciclo izquierdo con el derecho temporalmente más cercano.
    Evita mostrar por defecto ciclos que pertenecen a instantes muy diferentes.
    """
    pairs = []
    used_r = set()
    for li, lc in enumerate(left_cycles or []):
        lmid = (lc["ic_frame"] + lc["next_ic_frame"]) / 2.0
        best = None
        for ri, rc in enumerate(right_cycles or []):
            rmid = (rc["ic_frame"] + rc["next_ic_frame"]) / 2.0
            overlap = max(0, min(lc["next_ic_frame"], rc["next_ic_frame"]) - max(lc["ic_frame"], rc["ic_frame"]))
            dist = abs(lmid-rmid)
            score = dist - 1.5*overlap
            if best is None or score < best[0]:
                best = (score, ri)
        pairs.append((li, best[1] if best else None))

    if not pairs and right_cycles:
        for ri in range(len(right_cycles)):
            pairs.append((None, ri))
    return pairs


def _make_cycle_snapshot(seg, fps):
    """v0.10.9 · snapshot con cadena temporal validada."""
    if seg is None or not isinstance(seg,pd.DataFrame) or seg.empty:
        return None
    snap_seg=seg.copy().reset_index(drop=True)
    lmask,_,_=_support_mask_2d(snap_seg,"L",float(fps))
    rmask,_,_=_support_mask_2d(snap_seg,"R",float(fps))
    L0=_support_cycle_summary(lmask,float(fps))
    R0=_support_cycle_summary(rmask,float(fps))
    Lp,Rp,event_source,event_info=_select_temporal_event_chain(snap_seg,float(fps),L0,R0)
    L=_support_summary_positions_to_video_frames(Lp,snap_seg)
    R=_support_summary_positions_to_video_frames(Rp,snap_seg)
    return {
        "seg":snap_seg,"fps":float(fps),"left":L,"right":R,
        "event_source":event_source,
        "event_quality":event_info.get("quality","") if isinstance(event_info,dict) else "",
        "event_notes":event_info.get("reason","") if isinstance(event_info,dict) else "",
    }


def _video_frame_from_bytes(video_bytes, frame_no):
    """
    Extrae un frame de un MP4 guardado en session_state sin persistirlo.
    Usa un archivo temporal efímero solo durante la lectura.
    """
    if not video_bytes:
        return None
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
            tmp.write(video_bytes)
            tmp_path = tmp.name
        cap = cv2.VideoCapture(tmp_path)
        if not cap.isOpened():
            return None
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_no))
        ok, frame = cap.read()
        cap.release()
        if not ok or frame is None:
            return None
        return frame
    finally:
        if tmp_path:
            try:
                Path(tmp_path).unlink(missing_ok=True)
            except Exception:
                pass


def _encode_phase_overlay(frame, frame_no, left_cycle, right_cycle):
    if frame is None:
        return None
    lines = []
    for cyc, label in [(left_cycle, "IZQ"), (right_cycle, "DER")]:
        info = _cycle_phase_for_frame(cyc, int(frame_no)) if cyc else None
        if info:
            pct, phase = info
            lines.append(f"{label}: {phase} · {pct:.1f}%")
    if not lines:
        lines = ["Fuera del ciclo seleccionado"]

    y = 35
    for line in lines:
        cv2.putText(frame, line, (20,y), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (0,0,0), 4, cv2.LINE_AA)
        cv2.putText(frame, line, (20,y), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (255,255,255), 2, cv2.LINE_AA)
        y += 32

    ok, enc = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY),92])
    return enc.tobytes() if ok else None


def _phase_name_from_pct(pct, stance_pct=60.0):
    """
    Conventional descriptive subdivision of the gait cycle.
    Percentages are contextual, not force-plate validated.
    """
    if not np.isfinite(pct):
        return "No calculable"
    pct = float(pct) % 100.0
    stance_pct = float(np.clip(stance_pct if np.isfinite(stance_pct) else 60.0, 45.0, 80.0))

    # Split stance into loading / mid / terminal / pre-swing proportionally.
    loading_end = min(12.0, stance_pct * 0.20)
    mid_end = max(loading_end + 8.0, stance_pct * 0.55)
    terminal_end = max(mid_end + 8.0, stance_pct * 0.85)
    if pct < loading_end:
        return "Respuesta a la carga"
    if pct < mid_end:
        return "Apoyo medio"
    if pct < terminal_end:
        return "Apoyo terminal"
    if pct < stance_pct:
        return "Pre-oscilación / propulsión"

    swing_span = max(1.0, 100.0 - stance_pct)
    swing_rel = (pct - stance_pct) / swing_span
    if swing_rel < 0.33:
        return "Swing inicial"
    if swing_rel < 0.67:
        return "Swing medio"
    return "Swing terminal"



def _summary_from_positions(ic_positions, to_positions, next_ic_positions, fps, source="robust_distal"):
    ic=np.asarray(ic_positions,dtype=int); to=np.asarray(to_positions,dtype=int); nxt=np.asarray(next_ic_positions,dtype=int)
    n=min(len(ic),len(to),len(nxt)); ic,to,nxt=ic[:n],to[:n],nxt[:n]
    keep=(nxt>ic)&(to>ic)&(to<nxt); ic,to,nxt=ic[keep],to[keep],nxt[keep]
    cycle=(nxt-ic)/float(fps) if len(ic) else np.asarray([],float)
    stance=(to-ic)/float(fps) if len(ic) else np.asarray([],float)
    swing=(nxt-to)/float(fps) if len(ic) else np.asarray([],float)
    stance_pct=np.where(cycle>0,stance/cycle*100.0,np.nan) if len(ic) else np.asarray([],float)
    score=90.0 if len(cycle)>=3 else (68.0 if len(cycle)==2 else 40.0)
    quality="Alta" if score>=75 else ("Moderada" if score>=55 else "No fiable")
    return {
        "stance":stance,"swing":swing,"cycle":cycle,"stance_pct":stance_pct,
        "swing_pct":100.0-stance_pct if len(stance_pct) else np.asarray([],float),
        "ic":ic,"to":to,"next_ic":nxt,"starts":ic.copy(),"ends":to.copy(),
        "quality":quality,"score":float(score),"cycle_error_pct":np.nan,
        "fragmentation_index":np.nan,"reason":"cadena robusta distal",
        "event_source":source,"coverage_mask":np.zeros(0,dtype=bool),
    }


def _robust_distal_support_summaries(seg, fps):
    """v0.10.9 · IC/TO alternantes con QC temporal amplio."""
    empty=_summary_from_positions([],[],[],fps)
    if seg is None or len(seg)<max(45,int(round(2.5*fps))):
        return empty,empty,{"quality":"No fiable","reason":"segmento insuficiente"}

    need=["LAnkle_y","LHeel_y","LBigToe_y","RAnkle_y","RHeel_y","RBigToe_y",
          "LHip_y","RHip_y","LShoulder_y","RShoulder_y"]
    if not all(c in seg.columns for c in need):
        return empty,empty,{"quality":"No calculable","reason":"faltan trayectorias distales"}

    def a(c): return pd.to_numeric(seg[c],errors="coerce").to_numpy(float)
    ly=(a("LAnkle_y")+a("LHeel_y")+a("LBigToe_y"))/3.0
    ry=(a("RAnkle_y")+a("RHeel_y")+a("RBigToe_y"))/3.0
    torso=np.abs((a("LHip_y")+a("RHip_y"))/2.0-(a("LShoulder_y")+a("RShoulder_y"))/2.0)

    def clean(x):
        s=pd.Series(x).replace([np.inf,-np.inf],np.nan).interpolate(limit_direction="both")
        return s.to_numpy(float) if s.notna().sum()>=max(20,int(0.7*len(s))) else None
    ly,ry,torso=clean(ly),clean(ry),clean(torso)
    if ly is None or ry is None or torso is None:
        return empty,empty,{"quality":"No calculable","reason":"tracking distal insuficiente"}

    floor=np.nanpercentile(torso[np.isfinite(torso)],20) if np.isfinite(torso).any() else 1.0
    sig=(ry-ly)/np.maximum(torso,max(floor,1e-6))
    ws=max(5,int(round(0.18*fps))|1); wl=max(ws+2,int(round(1.8*fps))|1)
    sig=pd.Series(sig).rolling(ws,center=True,min_periods=1).median().to_numpy(float)
    sig=sig-pd.Series(sig).rolling(wl,center=True,min_periods=1).mean().to_numpy(float)

    amp=np.nanmedian(np.abs(sig-np.nanmedian(sig)))*1.4826
    sd=np.nanstd(sig)
    if not np.isfinite(amp) or amp<=1e-8: amp=sd
    if not np.isfinite(amp) or amp<=1e-8:
        return empty,empty,{"quality":"No calculable","reason":"sin periodicidad distal"}

    min_dist=max(2,int(round(0.28*fps))); prom=max(0.42*amp,0.28*sd)
    pos,_=find_peaks(sig,distance=min_dist,prominence=prom)
    neg,_=find_peaks(-sig,distance=min_dist,prominence=prom)
    cand=[(int(i),"R",abs(float(sig[i]))) for i in pos]+[(int(i),"L",abs(float(sig[i]))) for i in neg]
    cand.sort(key=lambda z:z[0])

    collapsed=[]
    for ev in cand:
        if collapsed and collapsed[-1][1]==ev[1]:
            if ev[2]>collapsed[-1][2]: collapsed[-1]=ev
        else:
            collapsed.append(ev)

    alt=[]
    for ev in collapsed:
        if not alt:
            alt.append(ev); continue
        dt=(ev[0]-alt[-1][0])/float(fps)
        if dt<0.30:
            if ev[2]>alt[-1][2]: alt[-1]=ev
            continue
        if ev[1]==alt[-1][1]:
            if ev[2]>alt[-1][2]: alt[-1]=ev
            continue
        alt.append(ev)

    if len(alt)<5:
        return empty,empty,{"quality":"No fiable","reason":f"solo {len(alt)} eventos alternantes"}

    dts=np.diff([e[0] for e in alt])/float(fps)
    finite=dts[np.isfinite(dts)&(dts>=0.30)&(dts<=1.80)]
    if len(finite)<3:
        return empty,empty,{"quality":"No fiable","reason":"intervalos insuficientes"}
    med=float(np.nanmedian(finite)); mad=float(np.nanmedian(np.abs(finite-med)))
    lo=max(0.30,med-max(0.18,3.5*mad)); hi=min(1.80,med+max(0.25,3.5*mad))
    cleaned=[alt[0]]
    for ev in alt[1:]:
        dt=(ev[0]-cleaned[-1][0])/float(fps)
        if lo<=dt<=hi and ev[1]!=cleaned[-1][1]:
            cleaned.append(ev)
    if len(cleaned)<5: cleaned=alt

    side_events={"L":[e[0] for e in cleaned if e[1]=="L"],"R":[e[0] for e in cleaned if e[1]=="R"]}
    foot={"L":ly,"R":ry}

    def build(side):
        ics=side_events[side]; out_ic=[]; out_to=[]; out_next=[]; y=foot[side]
        for x0,x1 in zip(ics[:-1],ics[1:]):
            cyc=(x1-x0)/float(fps)
            if not (0.65<=cyc<=3.20): continue
            p0=max(x0+1,min(int(round(x0+0.45*(x1-x0))),x1-2))
            p1=max(p0+1,min(int(round(x0+0.80*(x1-x0))),x1-1))
            yy=y[p0:p1+1]
            if len(yy)<3 or not np.isfinite(yy).any(): continue
            ys=pd.Series(yy).rolling(max(3,int(round(0.10*fps))|1),center=True,min_periods=1).mean().to_numpy(float)
            to=p0+int(np.nanargmin(np.gradient(ys)))
            pct=(to-x0)/(x1-x0)*100.0
            if 45.0<=pct<=80.0:
                out_ic.append(x0); out_to.append(to); out_next.append(x1)
        return _summary_from_positions(out_ic,out_to,out_next,fps)

    L,R=build("L"),build("R")
    ncy=len(L.get("cycle",[]))+len(R.get("cycle",[]))
    q="Alta" if ncy>=6 else ("Moderada" if ncy>=4 else "No fiable")
    return L,R,{"quality":q,"reason":f"{len(cleaned)} eventos alternantes; {ncy} ciclos robustos"}


def _summary_temporally_suspicious(summary):
    if not isinstance(summary,dict): return True
    cyc=np.asarray(summary.get("cycle",[]),float); pct=np.asarray(summary.get("stance_pct",[]),float)
    if len(cyc)<2 or len(pct)<2: return True
    good=np.isfinite(cyc)&np.isfinite(pct)&(cyc>=0.65)&(cyc<=3.20)&(pct>=45.0)&(pct<=80.0)
    return int(good.sum())<2


def _select_temporal_event_chain(seg,fps,L0,R0):
    Lr,Rr,info=_robust_distal_support_summaries(seg,fps)
    original_bad=_summary_temporally_suspicious(L0) or _summary_temporally_suspicious(R0)
    robust_good=(not _summary_temporally_suspicious(Lr)) and (not _summary_temporally_suspicious(Rr))
    if original_bad and robust_good:
        return Lr,Rr,"robust_distal",info
    return L0,R0,"support_mask",info


def _build_side_cycles(summary, fps, side):
    """
    v0.10.1 · Convierte el resumen IC→TO→nextIC en ciclos explícitos.
    Usa directamente los tres arrays validados por _support_cycle_summary,
    evitando asumir que ic contiene también el IC siguiente.
    """
    rows = []
    ics = np.asarray(summary.get("ic", []), dtype=int)
    tos = np.asarray(summary.get("to", []), dtype=int)
    next_ics = np.asarray(summary.get("next_ic", []), dtype=int)
    cycles = np.asarray(summary.get("cycle", []), dtype=float)
    stances = np.asarray(summary.get("stance", []), dtype=float)

    n = min(len(ics), len(tos), len(next_ics), len(cycles), len(stances))
    for i in range(n):
        ic = int(ics[i])
        to = int(tos[i])
        next_ic = int(next_ics[i])
        if not (ic < to <= next_ic):
            continue

        cycle_s = float((next_ic - ic) / fps) if next_ic > ic else np.nan
        stance_s = float((to - ic) / fps)
        swing_s = float((next_ic - to) / fps)
        stance_pct = float(stance_s / cycle_s * 100.0) if np.isfinite(cycle_s) and cycle_s > 0 else np.nan

        rows.append({
            "side": side,
            "cycle_index": len(rows) + 1,
            "ic_frame": ic,
            "to_frame": to,
            "next_ic_frame": next_ic,
            "cycle_s": cycle_s,
            "stance_s": stance_s,
            "swing_s": swing_s,
            "stance_pct": stance_pct,
        })
    return rows


def _cycle_phase_for_frame(cycle, frame):
    if not cycle:
        return None
    ic = cycle["ic_frame"]
    nic = cycle["next_ic_frame"]
    if frame < ic or frame > nic or nic <= ic:
        return None
    pct = (frame - ic) / (nic - ic) * 100.0
    phase = _phase_name_from_pct(pct, cycle.get("stance_pct", 60.0))
    return float(pct), phase


def _find_cycle_for_frame(cycles, frame):
    for c in cycles:
        if c["ic_frame"] <= frame <= c["next_ic_frame"]:
            return c
    return None


def _normalized_cycle_series(seg, cycle, columns):
    """
    v0.10.4 · Extrae el ciclo usando frames REALES y devuelve señal 0–100%.
    Si hay frames perdidos por tracking, interpola solo para visualización.
    """
    if cycle is None or seg is None or seg.empty:
        return pd.DataFrame()
    ic, nic = int(cycle["ic_frame"]), int(cycle["next_ic_frame"])
    if nic <= ic:
        return pd.DataFrame()

    part = seg[(pd.to_numeric(seg["frame"],errors="coerce") >= ic) &
               (pd.to_numeric(seg["frame"],errors="coerce") <= nic)].copy()
    if part.empty:
        return pd.DataFrame()

    part["cycle_pct"] = (
        (pd.to_numeric(part["frame"],errors="coerce") - ic) / float(nic-ic) * 100.0
    )
    keep=["frame","cycle_pct"]+[c for c in columns if c in part.columns]
    part=part[keep].sort_values("cycle_pct")

    for c in columns:
        if c in part.columns:
            part[c]=pd.to_numeric(part[c],errors="coerce")
            if part[c].notna().sum() >= 2:
                part[c]=part[c].interpolate(limit_direction="both")
    return part


def _compute_display_kinematics(seg):
    """
    v0.10.4 · Señales para el visor del ciclo.

    Prioriza las columnas cinemáticas YA calculadas por compute_metrics para
    evitar que la pestaña 9 reconstruya una señal distinta. Solo calcula
    aliases/fallbacks cuando una variable todavía no existe.
    """
    out = seg.copy()

    # V49 · asegurar que el visor/ciclo dispone de las mismas variables derivadas
    # que compute_metrics. En V48, cycle_seg podía contener solo landmarks HALPE26,
    # por lo que las curvas frontales quedaban en cero aunque las métricas frontales
    # sí existieran. Aquí se reconstruyen de forma determinista desde los mismos
    # puntos 2D antes de aplicar aliases.
    try:
        landmark_core = [
            "LHip_x","LHip_y","RHip_x","RHip_y",
            "LKnee_x","LKnee_y","RKnee_x","RKnee_y",
            "LAnkle_x","LAnkle_y","RAnkle_x","RAnkle_y",
            "LShoulder_x","LShoulder_y","RShoulder_x","RShoulder_y",
            "LHeel_x","LHeel_y","RHeel_x","RHeel_y",
            "LBigToe_x","LBigToe_y","RBigToe_x","RBigToe_y",
        ]
        if all(c in out.columns for c in landmark_core):
            if not any(c in out.columns for c in ("L_knee_flex","R_knee_flex","pelvis_obliquity","L_frontal_knee_dev")):
                out = add_angle_columns(out)
                out = add_frontal_columns(out)
            else:
                if not all(c in out.columns for c in ("L_knee_flex","R_knee_flex","L_hip_flex","R_hip_flex","L_ankle_angle","R_ankle_angle")):
                    out = add_angle_columns(out)
                if not all(c in out.columns for c in ("pelvis_obliquity","shoulder_obliquity","trunk_lateral_lean","L_frontal_knee_dev","R_frontal_knee_dev","L_foot_progress_proj","R_foot_progress_proj","L_rearfoot_tilt_proj","R_rearfoot_tilt_proj")):
                    out = add_frontal_columns(out)
    except Exception:
        # El visor nunca debe bloquear el análisis principal por un fallback gráfico.
        pass

    aliases = {
        "L_knee_flex": "Flexión rodilla Izquierda (°)",
        "R_knee_flex": "Flexión rodilla Derecha (°)",
        "L_hip_flex": "Flexión cadera Izquierda (°)",
        "R_hip_flex": "Flexión cadera Derecha (°)",
        "L_ankle_angle": "Ángulo tobillo Izquierda (°)",
        "R_ankle_angle": "Ángulo tobillo Derecha (°)",
        "pelvis_obliquity": "Oblicuidad pélvica (°)",
        "shoulder_obliquity": "Oblicuidad hombros (°)",
        "trunk_lateral_lean": "Inclinación tronco (°)",
        "L_frontal_knee_dev": "Desviación frontal rodilla Izquierda (°)",
        "R_frontal_knee_dev": "Desviación frontal rodilla Derecha (°)",
        "L_foot_progress_proj": "Orientación pie Izquierda (°)",
        "R_foot_progress_proj": "Orientación pie Derecha (°)",
        "L_rearfoot_tilt_proj": "Retropié Izquierda (°)",
        "R_rearfoot_tilt_proj": "Retropié Derecha (°)",
    }
    for src, label in aliases.items():
        if src in out.columns:
            out[label] = pd.to_numeric(out[src], errors="coerce")

    # Fallback rodilla si por cualquier motivo no existe la columna derivada.
    def xy(name):
        return out[f"{name}_x"].to_numpy(float), out[f"{name}_y"].to_numpy(float)

    for side_label, side, hip, knee, ankle in [
        ("Izquierda","L","LHip","LKnee","LAnkle"),
        ("Derecha","R","RHip","RKnee","RAnkle"),
    ]:
        label = f"Flexión rodilla {side_label} (°)"
        if label not in out.columns:
            need=[f"{p}_{a}" for p in (hip,knee,ankle) for a in ("x","y")]
            if all(c in out.columns for c in need):
                hx,hy=xy(hip); kx,ky=xy(knee); ax,ay=xy(ankle)
                v1=np.c_[hx-kx,hy-ky]; v2=np.c_[ax-kx,ay-ky]
                den=np.linalg.norm(v1,axis=1)*np.linalg.norm(v2,axis=1)
                dot=np.sum(v1*v2,axis=1)
                cosang=np.divide(dot,den,out=np.full(len(out),np.nan),where=den>1e-9)
                out[label]=180.0-np.degrees(np.arccos(np.clip(cosang,-1,1)))

    # Suavizado ligero SOLO para presentación; no cambia las métricas.
    display_cols=[c for c in out.columns if any(t in c for t in (
        "Flexión ","Ángulo tobillo","Oblicuidad ","Inclinación tronco",
        "Desviación frontal","Orientación pie","Retropié"
    ))]
    for c in display_cols:
        s=pd.to_numeric(out[c],errors="coerce")
        if s.notna().sum() >= 3:
            out[c]=s.interpolate(limit=3,limit_direction="both").rolling(3,center=True,min_periods=1).mean()

    return out


def _phase_segments_for_stance(stance_pct):
    stance = float(np.clip(stance_pct if np.isfinite(stance_pct) else 60.0, 35.0, 90.0))
    loading = min(12.0, 0.20*stance)
    mid = max(loading+5.0, 0.55*stance)
    terminal = max(mid+5.0, 0.85*stance)
    swing = 100.0-stance
    s1 = stance + swing/3.0
    s2 = stance + 2.0*swing/3.0
    return [
        (0, loading, "Carga"),
        (loading, mid, "Apoyo medio"),
        (mid, terminal, "Apoyo terminal"),
        (terminal, stance, "Propulsión"),
        (stance, s1, "Swing inicial"),
        (s1, s2, "Swing medio"),
        (s2, 100, "Swing terminal"),
    ]





def _macro_phase(label):
    """
    Simplifica la visualización global a tres bloques clínicamente legibles.
    El detalle fino sigue mostrándose en los indicadores de fase actual.
    """
    if label in ("Respuesta a la carga", "Carga", "Apoyo medio", "Apoyo terminal"):
        return "Apoyo"
    if label in ("Pre-oscilación / propulsión", "Propulsión"):
        return "Propulsión"
    if label in ("Swing inicial", "Swing medio", "Swing terminal"):
        return "Swing"
    return "Otro"


def _whole_segment_kinematic_figure(kin, variable_left, variable_right,
                                    fmin, fmax, fps, current_frame):
    """
    v0.10.8 · Curvas sobre el MISMO eje temporal (segundos) del mapa de fases.
    Esto elimina la falsa impresión de desalineación causada por comparar:
      mapa = tiempo total
      curva = 0–100% de un solo ciclo.
    """
    fig = plt.figure(figsize=(7.2, 2.55))
    # v0.10.11 · EXACTAMENTE el mismo rectángulo horizontal que el timeline.
    ax = fig.add_axes([0.095, 0.20, 0.875, 0.72])
    plotted = False

    x = (
        pd.to_numeric(kin["frame"], errors="coerce").to_numpy(float) - float(fmin)
    ) / float(fps)

    def add(var, label):
        nonlocal plotted
        if var not in kin.columns:
            return
        y = pd.to_numeric(kin[var], errors="coerce").to_numpy(float)
        ok = np.isfinite(x) & np.isfinite(y)
        if int(ok.sum()) >= 2:
            ax.plot(x[ok], y[ok], linewidth=1.5, label=label)
            plotted = True

    add(variable_left, f"Izq · {variable_left}")
    add(variable_right, f"Der · {variable_right}")

    duration = max(0.001, (float(fmax)-float(fmin))/float(fps))
    ax.set_xlim(0.0, duration)

    tcur = (float(current_frame)-float(fmin))/float(fps)
    ax.axvline(tcur, linewidth=2.0, label=f"Cursor {tcur:.2f} s")
    ax.set_xlabel("Tiempo del segmento (s)", fontsize=8)
    ax.set_ylabel("Magnitud 2D", fontsize=8)
    ax.tick_params(axis="both", labelsize=8)
    ax.set_title("Cinemática temporal · alineada con el mapa de fases", fontsize=10)
    ax.grid(True, alpha=0.20)
    if plotted:
        ax.legend(loc="best", fontsize=6.5, ncol=2)
    else:
        ax.text(
            0.5, 0.5, "Sin datos suficientes para las variables seleccionadas",
            transform=ax.transAxes, ha="center", va="center", fontsize=8.5
        )
    return fig


def _draw_phase_progress_overlay(frame, frame_no, left_cycles, right_cycles):
    """
    Overlay compacto para el vídeo reproducible:
    fase detallada + progreso de ciclo de cada extremidad.
    """
    if frame is None:
        return frame

    h, w = frame.shape[:2]
    entries = []
    for side, cycles in [("IZQ", left_cycles), ("DER", right_cycles)]:
        cyc = _find_cycle_for_frame(cycles, frame_no) if cycles else None
        info = _cycle_phase_for_frame(cyc, frame_no) if cyc else None
        if info:
            pct, phase = info
            entries.append((side, phase, pct))
        else:
            entries.append((side, "Entre ciclos", np.nan))

    # Fondo semitransparente superior.
    overlay = frame.copy()
    cv2.rectangle(overlay, (0,0), (w,78), (0,0,0), -1)
    frame = cv2.addWeighted(overlay, 0.45, frame, 0.55, 0)

    y = 26
    for side, phase, pct in entries:
        txt = f"{side}: {phase}" + (f" · {pct:.0f}%" if np.isfinite(pct) else "")
        cv2.putText(frame, txt, (12,y), cv2.FONT_HERSHEY_SIMPLEX, 0.58,
                    (255,255,255), 2, cv2.LINE_AA)
        if np.isfinite(pct):
            x0, x1 = 12, min(w-12, 240)
            yy = y + 9
            cv2.rectangle(frame, (x0,yy), (x1,yy+5), (100,100,100), -1)
            xf = int(round(x0 + (x1-x0)*float(np.clip(pct,0,100))/100.0))
            cv2.rectangle(frame, (x0,yy), (xf,yy+5), (255,255,255), -1)
        y += 35
    return frame


def _build_phase_review_video_bytes(source_bytes, left_cycles, right_cycles, fps):
    """
    Genera una vez por sesión un MP4 de revisión con fases/progreso embebidos.
    No se guarda en Supabase.
    """
    if not source_bytes:
        return None

    src_path = raw_path = out_path = None
    try:
        tmpdir = Path(tempfile.mkdtemp(prefix="gait_phase_review_"))
        src_path = tmpdir / "source.mp4"
        raw_path = tmpdir / "phase_raw.mp4"
        out_path = tmpdir / "phase_review.mp4"
        src_path.write_bytes(source_bytes)

        cap = cv2.VideoCapture(str(src_path))
        if not cap.isOpened():
            return None
        fps_in = float(cap.get(cv2.CAP_PROP_FPS) or fps or 30.0)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        writer = cv2.VideoWriter(
            str(raw_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps_in,
            (width, height),
        )
        frame_no = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame = _draw_phase_progress_overlay(
                frame, frame_no, left_cycles, right_cycles
            )
            writer.write(frame)
            frame_no += 1
        cap.release()
        writer.release()

        ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
        cmd = (
            f'"{ffmpeg}" -y -loglevel error -i "{raw_path}" '
            f'-c:v libx264 -pix_fmt yuv420p -movflags +faststart -an "{out_path}"'
        )
        rc = os.system(cmd)
        if rc == 0 and out_path.exists():
            return out_path.read_bytes()
        return raw_path.read_bytes() if raw_path.exists() else None
    finally:
        try:
            if 'tmpdir' in locals() and tmpdir.exists():
                shutil.rmtree(tmpdir, ignore_errors=True)
        except Exception:
            pass


def _change_video_speed_bytes(video_bytes, speed):
    """
    Cambia velocidad de reproducción manteniendo todos los frames.
    speed=0.5 -> cámara lenta 50%; speed=2 -> doble velocidad.
    """
    if not video_bytes:
        return None
    speed = float(speed)
    if speed <= 0:
        speed = 1.0
    if abs(speed-1.0) < 1e-9:
        return video_bytes

    try:
        tmpdir = Path(tempfile.mkdtemp(prefix="gait_speed_"))
        src = tmpdir / "src.mp4"
        out = tmpdir / "out.mp4"
        src.write_bytes(video_bytes)
        ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
        # setpts = PTS/speed
        cmd = (
            f'"{ffmpeg}" -y -loglevel error -i "{src}" '
            f'-filter:v "setpts=PTS/{speed:.6f}" '
            f'-c:v libx264 -pix_fmt yuv420p -movflags +faststart -an "{out}"'
        )
        rc = os.system(cmd)
        return out.read_bytes() if rc == 0 and out.exists() else None
    finally:
        try:
            if 'tmpdir' in locals() and tmpdir.exists():
                shutil.rmtree(tmpdir, ignore_errors=True)
        except Exception:
            pass


def _phase_intervals_for_cycle(cycle, fps, fmin):
    """
    Convierte un ciclo IC→TO→IC en intervalos de fase sobre TIEMPO REAL del vídeo.
    Devuelve (t0_s, t1_s, label).
    """
    if not cycle or cycle["next_ic_frame"] <= cycle["ic_frame"]:
        return []

    ic = float(cycle["ic_frame"])
    nic = float(cycle["next_ic_frame"])
    stance_pct = float(cycle.get("stance_pct", 60.0))
    out = []
    for a, b, label in _phase_segments_for_stance(stance_pct):
        fa = ic + (a/100.0)*(nic-ic)
        fb = ic + (b/100.0)*(nic-ic)
        out.append(((fa-fmin)/fps, (fb-fmin)/fps, label))
    return out


def _whole_segment_phase_timeline(left_cycles, right_cycles, fps, fmin, fmax, current_frame):
    """
    v0.10.10 · Timeline limpio, con leyenda separada del eje X y cursor maestro.
    """
    duration = max(0.001, (float(fmax)-float(fmin))/float(fps))
    fig = plt.figure(figsize=(7.2, 2.05))
    # v0.10.11 · rectángulo de ejes FIJO y compartido con la cinemática.
    ax = fig.add_axes([0.095, 0.28, 0.875, 0.60])
    ax.set_xlim(0, duration)
    ax.set_ylim(-0.15, 1.65)
    ax.set_yticks([1.08, 0.42])
    ax.set_yticklabels(["Izq.", "Der."])
    ax.set_xlabel("Tiempo del segmento (s)", fontsize=8, labelpad=8)
    ax.tick_params(axis="both", labelsize=8)
    ax.set_title("Fases de marcha · línea temporal", fontsize=10, pad=8)

    cycle_colors = plt.rcParams["axes.prop_cycle"].by_key().get("color", ["C0","C1","C2"])
    phase_color = {
        "Apoyo": cycle_colors[0 % len(cycle_colors)],
        "Propulsión": cycle_colors[1 % len(cycle_colors)],
        "Swing": cycle_colors[2 % len(cycle_colors)],
    }

    legend_seen = set()
    for cycles, y in [(left_cycles,1.08),(right_cycles,0.42)]:
        for cyc in cycles or []:
            raw = _phase_intervals_for_cycle(cyc, fps, fmin)
            grouped = []
            for t0,t1,label in raw:
                macro = _macro_phase(label)
                if grouped and grouped[-1][2] == macro and abs(grouped[-1][1]-t0) < 1e-6:
                    grouped[-1] = (grouped[-1][0], t1, macro)
                else:
                    grouped.append((t0,t1,macro))

            for t0,t1,macro in grouped:
                a=max(0.0,t0); b=min(duration,t1)
                if b<=a:
                    continue
                label_for_legend = macro if macro not in legend_seen else None
                ax.barh(
                    y,b-a,left=a,height=0.46,
                    alpha=0.55, edgecolor="white", linewidth=0.6,
                    color=phase_color.get(macro),
                    label=label_for_legend
                )
                legend_seen.add(macro)

            tic=(float(cyc["ic_frame"])-float(fmin))/float(fps)
            if 0<=tic<=duration:
                ax.vlines(tic,y-0.28,y+0.28,linewidth=0.55,alpha=0.45)

    tcur=(float(current_frame)-float(fmin))/float(fps)
    if 0<=tcur<=duration:
        ax.axvline(tcur,linewidth=2.2,zorder=20)
        ax.text(
            tcur,1.50,f"{tcur:.2f} s",
            ha="center",va="bottom",fontsize=7,
            bbox=dict(boxstyle="round,pad=0.15",fc="white",ec="none",alpha=0.85)
        )

    if legend_seen:
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(
            handles,labels,
            loc="upper center",
            bbox_to_anchor=(0.5,-0.34),
            ncol=3,frameon=False,fontsize=7,
            borderaxespad=0.0,columnspacing=1.8,handlelength=2.4
        )

    return fig



def _combined_phase_kinematic_figure(left_cycles, right_cycles, kin,
                                     variable_left, variable_right,
                                     fps, fmin, fmax, current_frame):
    """
    v0.10.12 · Un único lienzo con dos ejes que comparten exactamente X.
    Esto elimina cualquier diferencia de renderizado entre dos st.pyplot separados.
    """
    duration = max(0.001, (float(fmax)-float(fmin))/float(fps))
    tcur = (float(current_frame)-float(fmin))/float(fps)

    fig = plt.figure(figsize=(7.2, 4.85))

    # MISMO left y width en ambos ejes.
    left, width = 0.095, 0.875
    ax_phase = fig.add_axes([left, 0.65, width, 0.245])
    ax_kin = fig.add_axes([left, 0.13, width, 0.40], sharex=ax_phase)

    # ---------------------------
    # Panel superior: fases
    # ---------------------------
    ax_phase.set_xlim(0.0, duration)
    ax_phase.set_ylim(-0.15, 1.65)
    ax_phase.set_yticks([1.08, 0.42])
    ax_phase.set_yticklabels(["Izq.", "Der."])
    ax_phase.set_title("Fases de marcha · línea temporal", fontsize=10, pad=7)
    ax_phase.tick_params(axis="both", labelsize=8)
    ax_phase.tick_params(axis="x", labelbottom=False)

    cycle_colors = plt.rcParams["axes.prop_cycle"].by_key().get("color", ["C0","C1","C2"])
    phase_color = {
        "Apoyo": cycle_colors[0 % len(cycle_colors)],
        "Propulsión": cycle_colors[1 % len(cycle_colors)],
        "Swing": cycle_colors[2 % len(cycle_colors)],
    }

    legend_seen = set()
    for cycles, y in [(left_cycles,1.08),(right_cycles,0.42)]:
        for cyc in cycles or []:
            raw = _phase_intervals_for_cycle(cyc, fps, fmin)
            grouped = []
            for t0,t1,label in raw:
                macro = _macro_phase(label)
                if grouped and grouped[-1][2] == macro and abs(grouped[-1][1]-t0) < 1e-6:
                    grouped[-1] = (grouped[-1][0], t1, macro)
                else:
                    grouped.append((t0,t1,macro))

            for t0,t1,macro in grouped:
                a=max(0.0,t0); b=min(duration,t1)
                if b<=a:
                    continue
                legend_label = macro if macro not in legend_seen else None
                ax_phase.barh(
                    y,b-a,left=a,height=0.46,
                    alpha=0.55,edgecolor="white",linewidth=0.6,
                    color=phase_color.get(macro),label=legend_label
                )
                legend_seen.add(macro)

    # ---------------------------
    # Panel inferior: cinemática
    # ---------------------------
    x = (
        pd.to_numeric(kin["frame"], errors="coerce").to_numpy(float) - float(fmin)
    ) / float(fps)

    plotted=False
    def add(var,label):
        nonlocal plotted
        if var not in kin.columns:
            return
        y = pd.to_numeric(kin[var],errors="coerce").to_numpy(float)
        ok=np.isfinite(x)&np.isfinite(y)
        if int(ok.sum())>=2:
            ax_kin.plot(x[ok],y[ok],linewidth=1.45,label=label)
            plotted=True

    add(variable_left,f"Izq · {variable_left}")
    add(variable_right,f"Der · {variable_right}")

    ax_kin.set_xlim(0.0,duration)
    ax_kin.set_xlabel("Tiempo del segmento (s)",fontsize=8)
    ax_kin.set_ylabel("Magnitud 2D",fontsize=8)
    ax_kin.set_title("Cinemática temporal",fontsize=10,pad=7)
    ax_kin.tick_params(axis="both",labelsize=8)
    ax_kin.grid(True,alpha=0.20)

    if plotted:
        ax_kin.legend(loc="upper center",fontsize=6.5,ncol=2)

    # Un único cursor lógico dibujado en los dos ejes compartidos.
    ax_phase.axvline(tcur,linewidth=2.2,zorder=20)
    ax_kin.axvline(tcur,linewidth=2.2,zorder=20)

    # Etiqueta única arriba.
    ax_phase.text(
        tcur,1.50,f"{tcur:.2f} s",
        ha="center",va="bottom",fontsize=7,
        bbox=dict(boxstyle="round,pad=0.15",fc="white",ec="none",alpha=0.85)
    )

    # Leyenda de fases en el espacio entre paneles.
    if legend_seen:
        handles,labels=ax_phase.get_legend_handles_labels()
        ax_phase.legend(
            handles,labels,
            loc="upper center",
            bbox_to_anchor=(0.5,-0.18),
            ncol=3,frameon=False,fontsize=7,
            borderaxespad=0.0,columnspacing=1.8,handlelength=2.4
        )

    return fig


def _phase_context_text(cycles, frame):
    cyc = _find_cycle_for_frame(cycles, frame) if cycles else None
    if cyc is None:
        return "Entre ciclos", np.nan
    info = _cycle_phase_for_frame(cyc, frame)
    if info is None:
        return "Entre ciclos", np.nan
    return info[1], info[0]


def _all_event_frames(left_cycles, right_cycles):
    vals = []
    for side, cycles in [("I", left_cycles or []), ("D", right_cycles or [])]:
        for c in cycles:
            vals.append((int(c["ic_frame"]), f"{side} · IC"))
            if c.get("to_frame") is not None:
                vals.append((int(c["to_frame"]), f"{side} · TO"))
            vals.append((int(c["next_ic_frame"]), f"{side} · IC"))
    # Deduplicar por frame/label y ordenar.
    vals = sorted(set(vals), key=lambda x: (x[0], x[1]))
    return vals


def _nearest_cycle_for_frame(cycles, frame):
    if not cycles:
        return None
    inside = _find_cycle_for_frame(cycles, frame)
    if inside is not None:
        return inside
    return min(
        cycles,
        key=lambda c: abs(((c["ic_frame"]+c["next_ic_frame"])/2.0)-float(frame))
    )


def _phase_band_figure(cycle_left, cycle_right, current_frame=None):
    """
    v0.10.6 · Cada fila usa SU propio 0–100% y su propio cursor sincronizado
    al mismo frame de vídeo.
    """
    fig, ax = plt.subplots(figsize=(8.3, 1.55))
    ax.set_xlim(0,100)
    ax.set_ylim(-0.15,1.7)
    ax.set_yticks([1.12,0.38])
    ax.set_yticklabels(["Izq.","Der."])
    ax.set_xlabel("Ciclo (%)", fontsize=8)
    ax.tick_params(axis="both", labelsize=8)
    ax.set_title("Fases sincronizadas", fontsize=10)

    for cyc,y,label in [(cycle_left,1.12,"I"),(cycle_right,0.38,"D")]:
        stance=float(cyc.get("stance_pct",60.0)) if cyc else 60.0
        for a,b,name in _phase_segments_for_stance(stance):
            if b<=a:
                continue
            ax.barh(y,b-a,left=a,height=0.48,alpha=0.30,edgecolor="black",linewidth=0.35)
            if (b-a)>=9:
                ax.text((a+b)/2,y,name,ha="center",va="center",fontsize=6.0)
        if cyc:
            ax.axvline(stance,linewidth=0.9,linestyle="--")
            if current_frame is not None and cyc["next_ic_frame"]>cyc["ic_frame"]:
                pct=(float(current_frame)-cyc["ic_frame"])/(cyc["next_ic_frame"]-cyc["ic_frame"])*100.0
                if 0<=pct<=100:
                    ax.axvline(pct,linewidth=1.6,linestyle="-" if label=="I" else ":")
                    ax.text(pct,y+0.31,f"{label} {pct:.0f}%",ha="center",va="bottom",fontsize=6.5)

    fig.tight_layout(pad=0.5)
    return fig


def _kinematic_cycle_figure(df_left, df_right, variable_left=None, variable_right=None,
                              left_pct=None, right_pct=None):
    fig, ax = plt.subplots(figsize=(8.3,2.8))
    plotted=False

    def add(df,var,label):
        nonlocal plotted
        if df is None or df.empty or var not in df.columns:
            return
        x=pd.to_numeric(df["cycle_pct"],errors="coerce").to_numpy(float)
        y=pd.to_numeric(df[var],errors="coerce").to_numpy(float)
        ok=np.isfinite(x)&np.isfinite(y)
        if ok.sum()>=2:
            ax.plot(x[ok],y[ok],label=label,linewidth=1.8)
            plotted=True

    add(df_left,variable_left,f"Izq · {variable_left}")
    add(df_right,variable_right,f"Der · {variable_right}")

    if left_pct is not None and np.isfinite(left_pct) and 0<=left_pct<=100:
        ax.axvline(float(left_pct),linewidth=1.5,linestyle="-",label=f"Cursor I {left_pct:.0f}%")
    if right_pct is not None and np.isfinite(right_pct) and 0<=right_pct<=100:
        ax.axvline(float(right_pct),linewidth=1.5,linestyle=":",label=f"Cursor D {right_pct:.0f}%")

    ax.set_xlim(0,100)
    ax.set_xlabel("Ciclo (%)",fontsize=8)
    ax.set_ylabel("Magnitud 2D",fontsize=8)
    ax.tick_params(axis="both",labelsize=8)
    ax.set_title("Curva cinemática normalizada",fontsize=10)
    ax.grid(True,alpha=0.22)

    if plotted:
        ax.legend(loc="best",fontsize=6.5,ncol=2)
    else:
        ax.text(50,0.5,"Sin suficientes datos finitos para esta variable/ciclo",
                ha="center",va="center",transform=ax.get_xaxis_transform(),fontsize=8.5)
        ax.set_yticks([])
    fig.tight_layout(pad=0.6)
    return fig


def _video_frame_with_phase(video_path, frame_no, seg, left_cycle, right_cycle):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_no))
    ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        return None

    # Overlay keypoints/segments from tracking if row is available.
    if seg is not None and not seg.empty:
        row_match = seg.iloc[(seg["frame"]-int(frame_no)).abs().argsort()[:1]]
        if not row_match.empty:
            row = row_match.iloc[0]
            pairs = [
                ("LShoulder","RShoulder"),("LShoulder","LHip"),("RShoulder","RHip"),
                ("LHip","RHip"),("LHip","LKnee"),("LKnee","LAnkle"),
                ("RHip","RKnee"),("RKnee","RAnkle"),
            ]
            pts = {}
            for name in HALPE26:
                xk,yk,sk=f"{name}_x",f"{name}_y",f"{name}_score"
                if xk in row and yk in row and sk in row and np.isfinite(row[xk]) and np.isfinite(row[yk]) and float(row[sk])>=0.15:
                    pts[name]=(int(round(row[xk])),int(round(row[yk])))
            for a,b in pairs:
                if a in pts and b in pts:
                    cv2.line(frame,pts[a],pts[b],(255,255,255),2,cv2.LINE_AA)
            for p in pts.values():
                cv2.circle(frame,p,4,(255,255,255),-1,cv2.LINE_AA)

    lines=[]
    for cyc,label in [(left_cycle,"IZQ"),(right_cycle,"DER")]:
        info=_cycle_phase_for_frame(cyc,int(frame_no)) if cyc else None
        if info:
            pct,phase=info
            lines.append(f"{label}: {phase} · {pct:.1f}%")
    if not lines:
        lines=["Fuera del ciclo seleccionado"]

    y=35
    for line in lines:
        cv2.putText(frame,line,(20,y),cv2.FONT_HERSHEY_SIMPLEX,0.72,(0,0,0),4,cv2.LINE_AA)
        cv2.putText(frame,line,(20,y),cv2.FONT_HERSHEY_SIMPLEX,0.72,(255,255,255),2,cv2.LINE_AA)
        y+=32

    ok, enc = cv2.imencode(".jpg",frame,[int(cv2.IMWRITE_JPEG_QUALITY),92])
    return enc.tobytes() if ok else None


def _manual_event_editor(cycle, prefix):
    """
    Returns possibly corrected IC/TO/nextIC values using session_state.
    """
    if not cycle:
        return cycle
    ic_key=f"{prefix}_ic"
    to_key=f"{prefix}_to"
    nic_key=f"{prefix}_nic"
    if ic_key not in st.session_state:
        st.session_state[ic_key]=int(cycle["ic_frame"])
    if to_key not in st.session_state:
        st.session_state[to_key]=int(cycle["to_frame"]) if cycle.get("to_frame") is not None else int(round((cycle["ic_frame"]+cycle["next_ic_frame"])*0.6))
    if nic_key not in st.session_state:
        st.session_state[nic_key]=int(cycle["next_ic_frame"])

    a,b,c = st.columns(3)
    ic = a.number_input("IC", min_value=0, value=int(st.session_state[ic_key]), step=1, key=ic_key+"_widget")
    to = b.number_input("TO", min_value=0, value=int(st.session_state[to_key]), step=1, key=to_key+"_widget")
    nic = c.number_input("IC siguiente", min_value=0, value=int(st.session_state[nic_key]), step=1, key=nic_key+"_widget")
    if ic < to < nic:
        cycle = dict(cycle)
        cycle["ic_frame"]=int(ic)
        cycle["to_frame"]=int(to)
        cycle["next_ic_frame"]=int(nic)
        cycle["cycle_s"]=(nic-ic)  # se recalcula en segundos en la tabla usando fps_cycle
        cycle["stance_pct"]=(to-ic)/(nic-ic)*100.0
        st.session_state[ic_key]=int(ic)
        st.session_state[to_key]=int(to)
        st.session_state[nic_key]=int(nic)
    return cycle


APP_VERSION = "74"
TMP_ROOT = Path(tempfile.gettempdir()) / "physiosentinel_gait_online" / "sessions"
TMP_ROOT.mkdir(parents=True, exist_ok=True)

HALPE26 = {
    "Nose": 0,
    "LShoulder": 5, "RShoulder": 6,
    "LElbow": 7, "RElbow": 8,
    "LWrist": 9, "RWrist": 10,
    "LHip": 11, "RHip": 12,
    "LKnee": 13, "RKnee": 14,
    "LAnkle": 15, "RAnkle": 16,
    "Head": 17, "Neck": 18, "Hip": 19,
    "LBigToe": 20, "RBigToe": 21,
    "LSmallToe": 22, "RSmallToe": 23,
    "LHeel": 24, "RHeel": 25,
}
LOWER_BODY = ["LHip", "RHip", "LKnee", "RKnee", "LAnkle", "RAnkle", "LHeel", "RHeel", "LBigToe", "RBigToe"]
FOOT_POINTS = ["LAnkle", "RAnkle", "LHeel", "RHeel", "LBigToe", "RBigToe", "LSmallToe", "RSmallToe"]
UPPER_BODY = ["LShoulder", "RShoulder", "LElbow", "RElbow", "LWrist", "RWrist"]
ASSISTIVE_OPTIONS = ["Sin ayuda", "Bastón", "1 muleta", "2 muletas", "Caminador", "Rollator", "Otra"]
FUNCTIONAL_SUPPORT_OPTIONS = ["Independiente", "Supervisión", "Contacto ocasional con apoyo externo", "Apoyo externo frecuente", "Ayuda física de otra persona"]
SKELETON = [
    ("LShoulder", "RShoulder"), ("LShoulder", "LHip"), ("RShoulder", "RHip"), ("LHip", "RHip"),
    ("LShoulder", "LElbow"), ("LElbow", "LWrist"), ("RShoulder", "RElbow"), ("RElbow", "RWrist"),
    ("LHip", "LKnee"), ("LKnee", "LAnkle"), ("LAnkle", "LHeel"), ("LAnkle", "LBigToe"),
    ("RHip", "RKnee"), ("RKnee", "RAnkle"), ("RAnkle", "RHeel"), ("RAnkle", "RBigToe"),
    ("Neck", "Hip"), ("Nose", "Neck"),
]

st.set_page_config(page_title="PhysioSentinel Gait", page_icon="🚶", layout="wide")

# ------------------------- biblioteca de referencias -------------------------
# IMPORTANTE: estas bandas son referencias publicadas/contextuales, no umbrales diagnósticos.
# La comparabilidad depende de edad, sexo, velocidad, protocolo y sistema de medida.
REFERENCE_LIBRARY_VERSION = "2026.09.03"
REFERENCE_LIBRARY = {
    "cadence_exp": {
        "label": "Cadencia",
        "low": 114.95, "high": 118.35, "unit": "pasos/min",
        "population": "Adultos aparentemente sanos, marcha habitual al aire libre",
        "method": "Metaanálisis; media agrupada 116,65 pasos/min (IC95% 114,95–118,35)",
        "source": "Tudor-Locke et al., Sports Medicine 2021",
        "doi": "10.1007/s40279-020-01351-3",
        "applicability": "Contextual. El IC de la media no equivale a un intervalo individual de normalidad. No usar como umbral diagnóstico, especialmente en marcha neurológica."
    },
    "lateral_cadence_exp": {
        "label": "Cadencia lateral",
        "low": 114.95, "high": 118.35, "unit": "pasos/min",
        "population": "Adultos aparentemente sanos, marcha habitual al aire libre",
        "method": "Metaanálisis; media agrupada 116,65 pasos/min (IC95% 114,95–118,35)",
        "source": "Tudor-Locke et al., Sports Medicine 2021",
        "doi": "10.1007/s40279-020-01351-3",
        "applicability": "Contextual. El IC de la media no equivale a un intervalo individual de normalidad. No usar como umbral diagnóstico."
    },
}


# Referencias contextuales adicionales. Se usan solo cuando la definición de la métrica es comparable.
# Para proxies 2D sin equivalencia validada se muestra explícitamente que no existe un rango normativo transferible.
REFERENCE_CONTEXT = {
    "regularity_cv": "Referencia análoga: CV de tiempo de paso ~3.35% como umbral de variabilidad patológica en metaanálisis; no es idéntico a la alternancia 2D de esta app (König et al., 2019, PMID 31639377).",
    "temporal_asymmetry_exp": "Sin umbral universal transferible a este detector 2D. En laboratorio, la simetría del tiempo de apoyo suele ser cercana a 1:1; una cohorte joven femenina reportó ~0.7% a velocidad preferida (PMCID PMC6335661).",
    "stance_asymmetry_2d": "Referencia contextual: el tiempo de apoyo sano es aproximadamente simétrico entre lados; los límites dependen del índice y del sistema de medida. Este valor es una estimación 2D experimental.",
    "stance_pct_2d": "Referencia temporal clásica en marcha adulta confortable: apoyo ~60% del ciclo; contextual, no umbral diagnóstico y dependiente de velocidad/patología.",
    "swing_pct_2d": "Referencia temporal clásica en marcha adulta confortable: oscilación ~40% del ciclo; contextual, no umbral diagnóstico.",
    "double_support_pct_2d": "Referencia temporal clásica en marcha adulta confortable: doble apoyo total ~20–24% del ciclo (dos periodos de ~10–12%); aumenta habitualmente al disminuir la velocidad.",
    "double_support_time_2d": "Sin duración absoluta universal; depende de la duración del ciclo. Referencia porcentual contextual: ~20–24% del ciclo en marcha adulta confortable.",
    "swing_time_l_2d": "Sin duración absoluta universal; referencia proporcional contextual: oscilación ~40% del ciclo en marcha adulta confortable.",
    "swing_time_r_2d": "Sin duración absoluta universal; referencia proporcional contextual: oscilación ~40% del ciclo en marcha adulta confortable.",
    "swing_asymmetry_2d": "Sin umbral 2D universal validado para este detector. En marcha simétrica los tiempos de oscilación deberían aproximarse entre extremidades; interpretar longitudinalmente.",
    "initial_contact_foot_l_deg": "Sin rango normativo 2D universal transferible. La orientación en contacto inicial depende del plano de cámara, velocidad, calzado y estrategia de contacto.",
    "initial_contact_foot_r_deg": "Sin rango normativo 2D universal transferible. La orientación en contacto inicial depende del plano de cámara, velocidad, calzado y estrategia de contacto.",
    "initial_contact_rearfoot_l_deg": "Sin umbral 2D validado universal para inclinación del retropié en contacto inicial; descriptor proyectado.",
    "initial_contact_rearfoot_r_deg": "Sin umbral 2D validado universal para inclinación del retropié en contacto inicial; descriptor proyectado.",
    "loading_knee_l_deg": "Sin umbral 2D universal. Descriptor de estabilidad/alineación de rodilla durante la ventana de respuesta a la carga.",
    "loading_knee_r_deg": "Sin umbral 2D universal. Descriptor de estabilidad/alineación de rodilla durante la ventana de respuesta a la carga.",
    "terminal_foot_l_deg": "Sin rango normativo 2D universal transferible para orientación distal en pre-oscilación/despegue.",
    "terminal_foot_r_deg": "Sin rango normativo 2D universal transferible para orientación distal en pre-oscilación/despegue.",
    "pelvis_obliquity_rom": "No existe un rango normativo directamente transferible al ROM dinámico HALPE26 2D. La oblicuidad pélvica estática en población sana se ha descrito entre 0–5.6°, pero no equivale a este ROM dinámico (PMID 37254005).",
    "shoulder_obliquity_rom": "Sin rango normativo validado para este proxy 2D markerless; priorizar comparación intraindividual.",
    "trunk_lateral_lean_rom": "Sin rango normativo validado para este proxy 2D markerless; depende de velocidad, tarea y estrategia compensatoria.",
    "shoulder_pelvis_rel_rom": "Sin rango normativo validado para este acoplamiento 2D; usar como descriptor longitudinal.",
    "com_lateral_excursion_cm": "Sin rango universal: la excursión lateral del CoM depende de velocidad, ancho de paso y método. Solo se expresa en cm si existe escala espacial calibrada.",
    "com_lateral_excursion_norm_pct": "Sin rango universal validado para este proxy HALPE26. Normaliza la excursión mediolateral por la anchura pélvica aparente para facilitar comparaciones intraindividuales con geometría similar.",
    "com_velocity_ml_px_s": "Sin rango normativo universal en px/s. Descriptor longitudinal dependiente de la geometría de cámara; usar principalmente para comparación del mismo paciente/protocolo.",
    "com_bos_ratio_p95": "Sin umbral diagnóstico universal para este proxy 2D. 0 representa el centro entre ambos pies y valores absolutos próximos a 1 aproximan el centro de uno de los pies.",
    "com_bos_ratio_mean_abs": "Sin umbral diagnóstico universal para este proxy 2D; describe cuánto se aleja en promedio el centro corporal estimado del centro de la base de sustentación.",
    "step_to_step_similarity_index": "Índice interno experimental 0–100 derivado de la diferencia entre intervalos consecutivos de paso. 100 significa máxima repetibilidad temporal; no es una escala clínica validada.",
    "step_to_step_change_pct": "Sin umbral clínico universal para este detector 2D. Cuanto menor es el cambio porcentual entre pasos consecutivos, mayor es la consistencia temporal.",
    "bos_width_cm": "Sin rango universal transferible: el ancho de base depende de antropometría, edad y velocidad. Interpretar con contexto y basal individual.",
    "trendelenburg_drop_l_deg": "Sin umbral diagnóstico 2D universal. Descriptor proyectado de caída pélvica durante apoyo monopodal; confirmar clínicamente si es relevante.",
    "trendelenburg_drop_r_deg": "Sin umbral diagnóstico 2D universal. Descriptor proyectado de caída pélvica durante apoyo monopodal; confirmar clínicamente si es relevante.",
    "dynamic_knee_valgus_l_deg": "Sin rango normativo validado para HALPE26 2D. El valgo dinámico es multiplanar; este valor es solo desviación medial proyectada.",
    "dynamic_knee_valgus_r_deg": "Sin rango normativo validado para HALPE26 2D. El valgo dinámico es multiplanar; este valor es solo desviación medial proyectada.",
    "trunk_pelvis_coupling_r": "Sin banda normativa universal para este coeficiente 2D. Valores cercanos a +1 indican acoplamiento en fase y cercanos a -1, contrafase; el significado clínico depende de la tarea.",
    "trunk_pelvis_phase_deg": "Sin banda normativa universal para este desfase 2D. Interpretar longitudinalmente y junto con la calidad del tracking.",
}

def reference_text_for_metric(key):
    base = key
    for pref in ("front_", "lateral_"):
        if base.startswith(pref):
            base = base[len(pref):]
    ref = REFERENCE_LIBRARY.get(key) or REFERENCE_LIBRARY.get(base)
    if ref:
        return f"Referencia poblacional: {ref['low']:.2f}–{ref['high']:.2f} {ref['unit']} ({ref['population']}); contextual, no umbral diagnóstico."
    if base in REFERENCE_CONTEXT:
        return REFERENCE_CONTEXT[base]
    if any(tok in base for tok in ["knee_flex","hip_flex","ankle_angle","shoulder_elev","frontal_knee_dev","foot_progress","rearfoot_tilt","base_width_relative"]):
        return "Sin rango normativo validado directamente transferible a esta métrica 2D proyectada; usar comparación intraindividual y contexto clínico."
    if base in {"tracking_mean","good_frames_pct","foot_visibility_pct","upper_visibility_pct","step_count_consistency_error_pct"}:
        return "Referencia metodológica interna de calidad; no es una variable clínica normativa."
    if base in {"step_events_detected","segment_duration_s","expected_steps_from_cadence","cadence_count_segment","alternation_interval"}:
        return "Sin rango normativo clínico único; variable dependiente de la tarea/segmento y usada para consistencia interna."
    return "Sin rango normativo estandarizado validado para esta definición y método; interpretar respecto al basal individual y al contexto."

REFERENCE_SOURCES = [
    {"Fuente":"Kreusch et al. 2026","Uso":"Mapa de evidencia normativa en adultos sanos 18–65 años (105 estudios; 11.764 participantes)","DOI":"10.1016/j.gaitpost.2026.110176"},
    {"Fuente":"Herssens et al. 2018","Uso":"Parámetros espaciotemporales y variabilidad a lo largo de la vida adulta","DOI":"10.1016/j.gaitpost.2018.06.012"},
    {"Fuente":"Tudor-Locke et al. 2021","Uso":"Velocidad/cadencia de marcha habitual y otros ritmos en adultos sanos","DOI":"10.1007/s40279-020-01351-3"},
    {"Fuente":"Fukuchi et al. 2019","Uso":"Efecto de la velocidad sobre parámetros espaciotemporales, cinemática y cinética","DOI":"10.1186/s12984-019-0559-8"},
    {"Fuente":"Sato et al. 2023","Uso":"Referencias preliminares markerless de tronco y miembro inferior en mayores sanos japoneses","DOI":"10.1298/ptr.E10247"},
]

def reference_for_metric(key):
    return REFERENCE_LIBRARY.get(key)

def reference_position(value, ref):
    if ref is None or value is None or not np.isfinite(float(value)):
        return "Sin referencia"
    v=float(value)
    if v < ref["low"]: return "Por debajo de la banda publicada"
    if v > ref["high"]: return "Por encima de la banda publicada"
    return "Dentro de la banda publicada"

# ------------------------- seguridad / secretos -------------------------
def secret(name, default=None):
    # Streamlit Community Cloud: st.secrets
    try:
        value = st.secrets.get(name, None)
        if value not in (None, ""):
            return value
    except Exception:
        pass

    # Hugging Face Spaces / Docker: secrets como variables de entorno
    return os.getenv(name, default)

SUPABASE_URL = (secret("SUPABASE_URL", "") or "").rstrip("/")
SUPABASE_KEY = secret("SUPABASE_SERVICE_ROLE_KEY", "") or ""
APP_PASSWORD = secret("GAIT_APP_PASSWORD", "") or ""


def require_password():
    if not APP_PASSWORD:
        st.warning("⚠️ GAIT_APP_PASSWORD no está configurada. Modo de prueba sin control de acceso.")
        return True
    if st.session_state.get("authenticated"):
        return True
    st.title("PhysioSentinel Gait")
    st.caption("Acceso protegido")
    pwd = st.text_input("Contraseña", type="password")
    if st.button("Entrar", type="primary"):
        if pwd == APP_PASSWORD:
            st.session_state.authenticated = True
            st.rerun()
        else:
            st.error("Contraseña incorrecta.")
    return False

if not require_password():
    st.stop()

# ------------------------- Supabase REST -------------------------
def sb_ready():
    return bool(SUPABASE_URL and SUPABASE_KEY)


def sb_headers(extra_prefer=None):
    h = {
        "apikey": SUPABASE_KEY,
        "Authorization": f"Bearer {SUPABASE_KEY}",
        "Content-Type": "application/json",
    }
    if extra_prefer:
        h["Prefer"] = extra_prefer
    return h


def sb_request(method, table, params=None, payload=None, prefer=None, timeout=30):
    if not sb_ready():
        raise RuntimeError("Supabase no está configurado en Streamlit Secrets.")
    url = f"{SUPABASE_URL}/rest/v1/{table}"
    r = requests.request(method, url, headers=sb_headers(prefer), params=params, json=payload, timeout=timeout)
    if r.status_code >= 300:
        raise RuntimeError(f"Supabase {r.status_code}: {r.text[:800]}")
    if not r.text.strip():
        return None
    try:
        return r.json()
    except Exception:
        return r.text


def sb_upsert_patient(code):
    data = sb_request(
        "POST", "gait_patients",
        params={"on_conflict": "code"},
        payload={"code": code},
        prefer="resolution=merge-duplicates,return=representation",
    )
    if not data:
        data = sb_request("GET", "gait_patients", params={"code": f"eq.{code}", "select": "id,code"})
    return data[0]["id"]


def sb_create_session(code, record_name, mode, view, meta, assistive_device="Sin ayuda", frontal_orientation="No especificada", meta2=None, calibration_profile_name=None):
    patient_id = sb_upsert_patient(code)
    session_id = str(uuid.uuid4())
    payload = {
        "id": session_id,
        "patient_id": patient_id,
        "record_name": record_name,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "mode": mode,
        "view": view or "",
        "assistive_device": assistive_device or "Sin ayuda",
        "assisted_gait": bool((assistive_device or "Sin ayuda") != "Sin ayuda"),
        "frontal_orientation": frontal_orientation or "No especificada",
        "fps": float(meta.get("fps", 0)) if meta else None,
        "frames": int(meta.get("frames", 0)) if meta else None,
        "duration_s": float(meta.get("duration", 0)) if meta else None,
        "fps_cam2": float(meta2.get("fps", 0)) if meta2 else None,
        "frames_cam2": int(meta2.get("frames", 0)) if meta2 else None,
        "duration_cam2_s": float(meta2.get("duration", 0)) if meta2 else None,
        "calibration_profile_name": calibration_profile_name or None,
        "video_persisted": False,
        "app_version": APP_VERSION,
    }
    sb_request("POST", "gait_sessions", payload=payload, prefer="return=minimal")
    return session_id


def sb_save_metrics(session_id, metrics, start_s, end_s):
    payload = []
    for m in metrics:
        v = m.get("value")
        if v is not None:
            try:
                v = float(v)
                if not np.isfinite(v):
                    v = None
            except Exception:
                v = None
        payload.append({
            "session_id": session_id,
            "metric_key": m["key"],
            "metric_label": m["label"],
            "value": v,
            "unit": m.get("unit", ""),
            "quality": m.get("quality", ""),
            "notes": m.get("notes", ""),
        })
    sb_request(
        "POST", "gait_metrics",
        params={"on_conflict": "session_id,metric_key"},
        payload=payload,
        prefer="resolution=merge-duplicates,return=minimal",
        timeout=60,
    )
    sb_request(
        "PATCH", "gait_sessions",
        params={"id": f"eq.{session_id}"},
        payload={"segment_start_s": float(start_s), "segment_end_s": float(end_s), "analysis_status": "completed"},
        prefer="return=minimal",
    )


def sb_list_patients():
    if not sb_ready():
        return pd.DataFrame()
    data = sb_request("GET", "gait_patients", params={"select": "id,code,created_at", "order": "code.asc"}) or []
    df = pd.DataFrame(data)
    if df.empty:
        return df
    # V67: en bases antiguas puede haber más de una fila física para el mismo
    # código de paciente. La interfaz debe presentar un solo paciente lógico.
    if "code" in df.columns:
        df["code"] = df["code"].astype(str)
        df = df.sort_values("created_at" if "created_at" in df.columns else "code").drop_duplicates("code", keep="first")
    return df.reset_index(drop=True)


def sb_list_calibrations():
    if not sb_ready():
        return []
    try:
        return sb_request(
            "GET", "gait_calibrations",
            params={"select": "id,name,camera_count,notes,created_at", "order": "name.asc"},
        ) or []
    except Exception:
        return []


def sb_get_calibration(name):
    if not sb_ready() or not name:
        return None
    data = sb_request(
        "GET", "gait_calibrations",
        params={"name": f"eq.{name}", "select": "id,name,camera_count,content_toml,notes,created_at", "limit": 1},
    ) or []
    return data[0] if data else None


def sb_upsert_calibration(name, content_toml, notes="", camera_count=2):
    payload = {
        "name": name.strip(),
        "camera_count": int(camera_count),
        "content_toml": content_toml,
        "notes": notes or "",
    }
    data = sb_request(
        "POST", "gait_calibrations",
        params={"on_conflict": "name"},
        payload=payload,
        prefer="resolution=merge-duplicates,return=representation",
    )
    return data[0] if data else None


def sb_update_session_3d(session_id, **fields):
    if not session_id or not fields:
        return
    clean = {k: v for k, v in fields.items() if v is not None}
    if clean:
        sb_request(
            "PATCH", "gait_sessions",
            params={"id": f"eq.{session_id}"},
            payload=clean,
            prefer="return=minimal",
        )


def sb_update_session_identity(session_id, patient_code=None, record_name=None):
    """
    v0.9.3 · Actualiza únicamente campos ya existentes en Supabase.
    Edad/sexo/fecha clínica permanecen en la sesión local y en el informe
    hasta que el esquema remoto disponga de columnas específicas.
    """
    if not session_id:
        return
    payload = {}
    if patient_code:
        patient_id = sb_upsert_patient(str(patient_code).strip())
        payload["patient_id"] = patient_id
    if record_name is not None:
        payload["record_name"] = str(record_name).strip()
    if payload:
        sb_request(
            "PATCH", "gait_sessions",
            params={"id": f"eq.{session_id}"},
            payload=payload,
            prefer="return=minimal",
        )


def sb_delete_session(session_id):
    """Elimina y VERIFICA un registro persistente de Supabase.

    V59: PostgREST puede responder 2xx a un DELETE aunque el filtro no haya
    eliminado filas. Por eso ya no se considera borrado por el mero código HTTP:
    se solicita la representación eliminada y después se comprueba que no quede
    ni la sesión ni métricas asociadas. El paciente se conserva.
    """
    sid = str(session_id or "").strip()
    if not sid:
        raise ValueError("Falta el identificador de la sesión.")

    # Primero las métricas, por compatibilidad con FK sin ON DELETE CASCADE.
    sb_request(
        "DELETE", "gait_metrics",
        params={"session_id": f"eq.{sid}"},
        prefer="return=representation",
    )
    deleted_sessions = sb_request(
        "DELETE", "gait_sessions",
        params={"id": f"eq.{sid}"},
        prefer="return=representation",
    )

    # Verificación remota explícita: evita mostrar un falso mensaje de éxito.
    remaining_session = sb_request(
        "GET", "gait_sessions",
        params={"id": f"eq.{sid}", "select": "id", "limit": 1},
    ) or []
    remaining_metrics = sb_request(
        "GET", "gait_metrics",
        params={"session_id": f"eq.{sid}", "select": "session_id", "limit": 1},
    ) or []
    if remaining_session or remaining_metrics:
        raise RuntimeError(
            "Supabase respondió a la solicitud, pero el registro sigue presente. "
            "No se marcará como eliminado."
        )
    return {
        "session_id": sid,
        "deleted_session_rows": len(deleted_sessions) if isinstance(deleted_sessions, list) else None,
        "verified": True,
    }


def sb_delete_patient(code):
    """Elimina un paciente y TODO su histórico de Supabase, con verificación.

    V66: elimina métricas -> sesiones -> paciente. Se consultan todas las filas
    que coincidan con el código para ser robustos incluso ante datos históricos
    duplicados. La operación sólo se considera correcta si la verificación final
    confirma que no quedan pacientes ni sesiones asociados.
    """
    patient_code = str(code or "").strip()
    if not patient_code:
        raise ValueError("Falta el código del paciente.")

    patients = sb_request(
        "GET", "gait_patients",
        params={"code": f"eq.{patient_code}", "select": "id,code"},
    ) or []
    if not patients:
        return {"patient_code": patient_code, "deleted_patients": 0, "deleted_sessions": 0, "verified": True}

    patient_ids = [str(row.get("id")) for row in patients if row.get("id")]
    session_ids = []
    for pid in patient_ids:
        rows = sb_request(
            "GET", "gait_sessions",
            params={"patient_id": f"eq.{pid}", "select": "id"},
        ) or []
        session_ids.extend(str(row.get("id")) for row in rows if row.get("id"))

    # Eliminar cada sesión con la misma verificación usada por el borrado individual.
    for sid in dict.fromkeys(session_ids):
        sb_delete_session(sid)

    deleted_patients = 0
    for pid in patient_ids:
        deleted = sb_request(
            "DELETE", "gait_patients",
            params={"id": f"eq.{pid}"},
            prefer="return=representation",
        )
        if isinstance(deleted, list):
            deleted_patients += len(deleted)

    # Verificación final independiente del mensaje HTTP de PostgREST.
    remaining_patients = sb_request(
        "GET", "gait_patients",
        params={"code": f"eq.{patient_code}", "select": "id"},
    ) or []
    remaining_sessions = []
    for pid in patient_ids:
        rows = sb_request(
            "GET", "gait_sessions",
            params={"patient_id": f"eq.{pid}", "select": "id", "limit": 1},
        ) or []
        remaining_sessions.extend(rows)

    if remaining_patients or remaining_sessions:
        raise RuntimeError(
            "Supabase no confirmó el borrado completo del paciente. "
            "La operación se ha detenido para evitar mostrar un falso éxito."
        )
    return {
        "patient_code": patient_code,
        "deleted_patients": deleted_patients,
        "deleted_sessions": len(set(session_ids)),
        "verified": True,
    }


def sb_patient_history(code):
    if not sb_ready():
        return pd.DataFrame()
    patient_code = str(code or "").strip()
    if not patient_code:
        return pd.DataFrame()

    # V67: recuperar TODAS las filas físicas que correspondan al mismo código.
    # Versiones antiguas pudieron dejar pacientes duplicados en Supabase; usar
    # limit=1 hacía que, tras borrar un registro, otro duplicado pudiera volver
    # a aparecer al reconstruir la pantalla.
    p = sb_request(
        "GET", "gait_patients",
        params={"code": f"eq.{patient_code}", "select": "id,code", "order": "created_at.asc"},
    ) or []
    patient_ids = [str(row.get("id")) for row in p if row.get("id")]
    if not patient_ids:
        return pd.DataFrame()

    pid_filter = f"in.({','.join(patient_ids)})"
    sessions = sb_request(
        "GET", "gait_sessions",
        params={"patient_id": pid_filter, "select": "id,patient_id,created_at,record_name,mode,view,assistive_device,assisted_gait,frontal_orientation,fps,frames,duration_s,fps_cam2,frames_cam2,duration_cam2_s,sync_offset_s,sync_correlation,sync_quality,calibration_profile_name,ready_3d,segment_start_s,segment_end_s,analysis_status", "order": "created_at.asc"},
    ) or []
    if not sessions:
        return pd.DataFrame()

    # IDs UUID seguros; PostgREST acepta in.(uuid1,uuid2,...).
    ids = ",".join(str(row["id"]) for row in sessions if row.get("id"))
    metrics = sb_request(
        "GET", "gait_metrics",
        params={"session_id": f"in.({ids})", "select": "session_id,metric_key,metric_label,value,unit,quality,notes"},
    ) or []
    sdf = pd.DataFrame(sessions)
    mdf = pd.DataFrame(metrics)
    if mdf.empty:
        # Mantener visibles registros sin métricas para poder borrarlos también.
        for col in ["session_id","metric_key","metric_label","value","unit","quality","notes"]:
            if col not in sdf.columns:
                sdf[col] = np.nan
        sdf["session_id"] = sdf["id"]
        return sdf
    return sdf.merge(mdf, left_on="id", right_on="session_id", how="left")

# ------------------------- utilidades vídeo -------------------------
def safe_name(text):
    text = (text or "sesion").strip()
    return re.sub(r"[^A-Za-z0-9_-]+", "_", text)[:60] or "sesion"


def video_metadata(path):
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return None
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    return {
        "fps": fps, "frames": frames, "width": width, "height": height,
        "duration": frames / fps if fps > 0 else 0,
        "orientation": "Vertical" if height > width else "Horizontal",
    }


def save_upload(uploaded, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        f.write(uploaded.getbuffer())


def create_temp_session(patient, record):
    old = st.session_state.get("session_dir")
    if old:
        try:
            shutil.rmtree(old, ignore_errors=True)
        except Exception:
            pass
    folder = TMP_ROOT / f"{safe_name(patient)}_{safe_name(record)}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
    (folder / "videos").mkdir(parents=True, exist_ok=True)
    return folder


def prepare_config(session_dir, multi_person=False, det_frequency=4):
    cfg = {
        "project": {
            "project_dir": str(session_dir),
            "multi_person": bool(multi_person),
            "participant_height": "auto",
            "participant_mass": 70,
            "frame_rate": "auto",
            "frame_range": "auto",
        },
        "pose": {
            "pose_model": "Body_with_feet",
            "mode": "balanced",
            "det_frequency": int(det_frequency),
            "device": "auto",
            "backend": "auto",
            "display_detection": False,
            "overwrite_pose": True,
            "save_video": "to_video",
            "output_format": "openpose",
            "tracking_mode": "sports2d",
        },
    }
    path = session_dir / "Config.toml"
    with open(path, "w", encoding="utf-8") as f:
        toml.dump(cfg, f)
    return path


# V65 · exclusión mutua para Pose2Sim/RTMPose.
# Streamlit puede mantener varias sesiones/pestañas activas a la vez. Como el backend
# de producción trabaja en CPU/OpenVINO, dos estimaciones simultáneas compiten por la
# misma CPU y pueden hacer que una sola cámara tarde más que un análisis biplanar.
# El lock es global al proceso/instancia y evita arrancar un segundo Pose2Sim mientras
# ya existe otro análisis pesado en curso.
POSE_LOCK_PATH = TMP_ROOT / ".physiosentinel_pose_estimation.lock"
POSE_LOCK_STALE_S = 6 * 60 * 60


def _pose_lock_info():
    try:
        if not POSE_LOCK_PATH.exists():
            return False, None
        age = max(0.0, time.time() - POSE_LOCK_PATH.stat().st_mtime)
        if age > POSE_LOCK_STALE_S:
            try:
                POSE_LOCK_PATH.unlink(missing_ok=True)
            except Exception:
                pass
            return False, None
        try:
            data = json.loads(POSE_LOCK_PATH.read_text(encoding="utf-8"))
        except Exception:
            data = {}
        data["age_s"] = age
        return True, data
    except Exception:
        return False, None


def _acquire_pose_lock(session_dir=None):
    token = uuid.uuid4().hex
    payload = {
        "token": token,
        "pid": os.getpid(),
        "started_at": datetime.now().isoformat(timespec="seconds"),
        "session_dir": str(session_dir or ""),
    }
    for _ in range(2):
        try:
            fd = os.open(str(POSE_LOCK_PATH), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            try:
                os.write(fd, json.dumps(payload, ensure_ascii=False).encode("utf-8"))
            finally:
                os.close(fd)
            return token, None
        except FileExistsError:
            busy, info = _pose_lock_info()
            if busy:
                return None, info
        except Exception as exc:
            return None, {"error": str(exc)}
    return None, {"error": "No se pudo adquirir el bloqueo de análisis."}


def _release_pose_lock(token):
    try:
        if not POSE_LOCK_PATH.exists():
            return
        try:
            data = json.loads(POSE_LOCK_PATH.read_text(encoding="utf-8"))
        except Exception:
            data = {}
        if not data or data.get("token") == token:
            POSE_LOCK_PATH.unlink(missing_ok=True)
    except Exception:
        pass


def run_pose2sim(config_path):
    from Pose2Sim import Pose2Sim
    Pose2Sim.poseEstimation(str(config_path))


# V68 · preselección multipersona rápida + enmascarado dinámico del sujeto.
# Objetivo: no estimar pose completa de acompañantes/terapeutas durante todos los
# frames. Primero se analizan pocos frames representativos, el clínico elige al
# paciente y después se crea un vídeo de entrada con el entorno fuera de una ROI
# dinámica enmascarado. Las coordenadas permanecen en el sistema original porque
# NO se recorta ni reescala el frame.
def _sample_frame_indices(meta, n=9):
    total = int((meta or {}).get("frames", 0) or 0)
    if total <= 0:
        return []
    if total <= n:
        return list(range(total))
    # Evita extremos de grabación, donde suele haber preparación/giro.
    lo = max(0, int(round(total * 0.06)))
    hi = min(total - 1, int(round(total * 0.94)))
    return sorted(set(int(round(x)) for x in np.linspace(lo, hi, n)))


def _make_sparse_video(src_path, dst_path, frame_indices, fps_out=6.0):
    cap = cv2.VideoCapture(str(src_path))
    if not cap.isOpened():
        raise RuntimeError(f"No se puede abrir el vídeo para preselección: {src_path}")
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    dst_path = Path(dst_path); dst_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(dst_path), cv2.VideoWriter_fourcc(*"mp4v"), float(fps_out), (w, h))
    if not writer.isOpened():
        cap.release(); raise RuntimeError("No se pudo crear el vídeo corto de preselección.")
    written=[]
    try:
        for fr in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(fr))
            ok, frame = cap.read()
            if ok and frame is not None:
                writer.write(frame); written.append(int(fr))
    finally:
        writer.release(); cap.release()
    if not written:
        raise RuntimeError("No se pudo extraer ningún frame para la preselección rápida.")
    return written


def _selection_to_original_frame(selection, sample_frames):
    if not selection:
        return selection
    out = dict(selection)
    pre_idx = int(selection.get("frame", 0))
    out["prepass_frame"] = pre_idx
    if sample_frames:
        pre_idx = max(0, min(pre_idx, len(sample_frames)-1))
        out["frame"] = int(sample_frames[pre_idx])
    return out


def _tracked_sparse_boxes(pre_json_dir, selection, choice_label, sample_frames):
    """V72 · Trayectoria sparse robusta para construir la ROI del paciente elegido.

    No exige una trayectoria continua. Parte del sujeto seleccionado manualmente,
    propaga su identidad por los frames del prepass con un criterio deliberadamente
    permisivo y omite frames ambiguos. Si solo existe el frame ancla, devuelve al
    menos esa caja para que el análisis selectivo pueda arrancar en lugar de abortar.
    """
    if not selection or not choice_label:
        return []
    cand = next((c for c in selection.get("candidates", []) if c.get("label") == choice_label), None)
    if cand is None:
        return []
    paths = sorted(Path(pre_json_dir).glob("*.json"))
    if not paths or not sample_frames:
        return []
    anchor_idx = int(selection.get("prepass_frame", 0))
    anchor_idx = max(0, min(anchor_idx, len(paths)-1))
    anchor_people = _frame_people(paths[anchor_idx])
    anchor = next((x for x in anchor_people if int(x["person_index"]) == int(cand["person_index"])), None)
    if anchor is None:
        return []

    accepted={anchor_idx: anchor}

    def walk(indices):
        prev=anchor
        gap=0
        for pos in indices:
            people=_frame_people(paths[pos])
            if not people:
                gap += 1
                continue
            scored=[]
            for person in people:
                cost=_identity_cost(prev["desc"], person["desc"])
                # Mantener afinidad con la firma corporal original sin exigir
                # continuidad perfecta; el prepass solo define la ROI esperada.
                shp=_shape_distance(anchor["desc"], person["desc"])
                scale_ratio=max(float(person["desc"].get("scale",1.0)),1.0)/max(float(anchor["desc"].get("scale",1.0)),1.0)
                scale_pen=abs(float(np.log(max(scale_ratio,1e-6))))
                cost += 0.28*min(2.0,shp) + 0.10*scale_pen
                scored.append((float(cost), person))
            scored.sort(key=lambda z:z[0])
            best_cost,best=scored[0]
            second_cost=scored[1][0] if len(scored)>1 else np.inf
            margin=float(second_cost-best_cost)
            # En el prepass preferimos saltarnos un frame dudoso antes que seguir
            # al acompañante, pero no abortamos la trayectoria por ello.
            if best_cost > (1.55 if gap else 1.35):
                gap += 1; continue
            if len(scored)>1 and margin < 0.06 and best_cost > 0.25:
                gap += 1; continue
            accepted[pos]=best
            prev=best
            gap=0

    walk(range(anchor_idx+1,len(paths)))
    walk(range(anchor_idx-1,-1,-1))

    boxes=[]
    for idx,item in sorted(accepted.items()):
        if idx < 0 or idx >= len(sample_frames):
            continue
        b=np.asarray(item["desc"]["bbox"],float)
        boxes.append((int(sample_frames[idx]), float(b[0]), float(b[1]), float(b[2]), float(b[3])))

    # Fallback V72: nunca impedir el análisis solo porque el prepass no consiguió
    # una trayectoria sparse suficiente. La caja elegida manualmente sigue siendo
    # una referencia válida para inicializar la ROI.
    if not boxes:
        b=np.asarray(cand.get("bbox",[0,0,1,1]),float)
        original_frame=int(selection.get("frame", sample_frames[anchor_idx]))
        boxes=[(original_frame,float(b[0]),float(b[1]),float(b[2]),float(b[3]))]
    return sorted(boxes,key=lambda z:z[0])


def _resolve_fullrun_subject(json_dir, selection, choice_label, search_radius=10):
    """V70 · Vincula la elección visual del prepass con una persona del análisis completo.

    Las coordenadas del vídeo selectivo son las mismas que en el original. Se busca
    alrededor del frame de selección el candidato cuya posición y tamaño mejor
    coincidan con la caja elegida por el clínico. Devuelve frame ancla e índice de
    persona para iniciar el seguimiento de identidad bloqueada.
    """
    if not selection or not choice_label:
        raise RuntimeError("Falta la selección manual del paciente.")
    target = next((c for c in selection.get("candidates", []) if c.get("label") == choice_label), None)
    if target is None:
        raise RuntimeError("No se encontró el sujeto elegido en la preselección.")
    paths = sorted(Path(json_dir).glob("*.json"))
    if not paths:
        raise RuntimeError("No hay JSON del análisis completo para bloquear la identidad.")

    target_frame = int(selection.get("frame", 0))
    tc = np.asarray([float(target.get("center_x", 0.0)), float(target.get("center_y", 0.0))], float)
    tb = np.asarray(target.get("bbox", [0,0,1,1]), float)
    tw=max(1.0,float(tb[2]-tb[0])); th=max(1.0,float(tb[3]-tb[1]))
    tscale=max(tw,th,1.0)

    candidates=[]
    for p in paths:
        fn=int(parse_frame_number(p,0))
        if abs(fn-target_frame) > int(search_radius):
            continue
        for item in _frame_people(p):
            d=item["desc"]
            cc=np.asarray(d["center"],float)
            cb=np.asarray(d["bbox"],float)
            cw=max(1.0,float(cb[2]-cb[0])); ch=max(1.0,float(cb[3]-cb[1]))
            center_cost=float(np.linalg.norm(cc-tc)/tscale)
            size_cost=abs(float(np.log(max(cw,ch,1.0)/tscale)))
            frame_cost=0.025*abs(fn-target_frame)
            cost=0.72*center_cost+0.23*size_cost+frame_cost
            candidates.append((cost,fn,int(item["person_index"]),center_cost,size_cost))
    if not candidates:
        raise RuntimeError("El paciente elegido no aparece con pose utilizable cerca del frame ancla.")
    candidates.sort(key=lambda x:x[0])
    best=candidates[0]
    if best[3] > 0.85:
        raise RuntimeError("No se pudo vincular con seguridad la selección visual al análisis completo.")
    return int(best[1]), int(best[2]), {
        "anchor_match_cost":float(best[0]),
        "anchor_center_cost":float(best[3]),
        "anchor_size_cost":float(best[4]),
        "target_frame":target_frame,
    }


def _build_masked_subject_video(src_path, dst_path, boxes, pad=0.48):
    """Enmascara fuera de una ROI interpolada sin cambiar resolución/coordenadas."""
    if not boxes:
        raise RuntimeError("No hay una referencia espacial del sujeto seleccionado para preparar el análisis selectivo.")
    cap=cv2.VideoCapture(str(src_path))
    if not cap.isOpened():
        raise RuntimeError(f"No se puede abrir {src_path}")
    fps=float(cap.get(cv2.CAP_PROP_FPS) or 25.0)
    total=int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    w=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); h=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    dst_path=Path(dst_path); dst_path.parent.mkdir(parents=True, exist_ok=True)
    writer=cv2.VideoWriter(str(dst_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w,h))
    if not writer.isOpened():
        cap.release(); raise RuntimeError("No se pudo crear el vídeo selectivo del paciente.")

    fr=np.asarray([b[0] for b in boxes],float)
    cx=np.asarray([(b[1]+b[3])/2 for b in boxes],float)
    cy=np.asarray([(b[2]+b[4])/2 for b in boxes],float)
    bw=np.asarray([max(40.0,b[3]-b[1]) for b in boxes],float)
    bh=np.asarray([max(80.0,b[4]-b[2]) for b in boxes],float)
    # V70 · ROI más selectiva. En V68/V69 una ROI demasiado ancha podía incluir al
    # acompañante y favorecer poses híbridas cuando ambos sujetos se solapaban.
    # Se mantiene margen suficiente para la zancada, sin recortar ni reescalar.
    min_w=max(0.24*w, float(np.nanmedian(bw))*1.55)
    min_h=max(0.48*h, float(np.nanmedian(bh))*1.35)
    try:
        i=0
        while True:
            ok, frame=cap.read()
            if not ok or frame is None:
                break
            cxi=float(np.interp(i,fr,cx,left=cx[0],right=cx[-1])); cyi=float(np.interp(i,fr,cy,left=cy[0],right=cy[-1]))
            bwi=max(min_w, float(np.interp(i,fr,bw,left=bw[0],right=bw[-1]))*(1.0+pad))
            bhi=max(min_h, float(np.interp(i,fr,bh,left=bh[0],right=bh[-1]))*(1.0+pad))
            x1=max(0,int(round(cxi-bwi/2))); x2=min(w,int(round(cxi+bwi/2)))
            y1=max(0,int(round(cyi-bhi/2))); y2=min(h,int(round(cyi+bhi/2)))
            masked=np.zeros_like(frame)
            if x2>x1 and y2>y1:
                masked[y1:y2,x1:x2]=frame[y1:y2,x1:x2]
            writer.write(masked); i+=1
    finally:
        writer.release(); cap.release()
    if total and i < max(1,int(total*0.9)):
        raise RuntimeError("El vídeo selectivo quedó incompleto.")
    return str(dst_path)


def _pose_completion_marker(session_dir):
    return Path(session_dir) / ".pose_v68_complete.json"


def _write_pose_completion_marker(session_dir, payload):
    try:
        _pose_completion_marker(session_dir).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception:
        pass


def find_pose_json_dir(session_dir, cam="cam01"):
    pose_dir = session_dir / "pose"
    if not pose_dir.exists():
        return None
    preferred = sorted([p for p in pose_dir.rglob(f"{cam}*_json") if p.is_dir()])
    if preferred:
        return preferred[0]
    candidates = sorted([p for p in pose_dir.rglob("*_json") if p.is_dir()])
    return candidates[0] if candidates else None


def parse_frame_number(path, fallback):
    m = re.search(r"(\d+)(?=\.json$)", path.name)
    return int(m.group(1)) if m else fallback


def load_pose_dataframe(json_dir):
    rows = []
    for i, path in enumerate(sorted(json_dir.glob("*.json"))):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            people = data.get("people", [])
            if not people:
                continue
            pts = people[0].get("pose_keypoints_2d", [])
            if len(pts) < 26 * 3:
                continue
            row = {"frame": parse_frame_number(path, i)}
            for name, idx in HALPE26.items():
                base = idx * 3
                row[f"{name}_x"] = float(pts[base])
                row[f"{name}_y"] = float(pts[base + 1])
                row[f"{name}_score"] = float(pts[base + 2])
            rows.append(row)
        except Exception:
            continue
    return pd.DataFrame(rows).sort_values("frame").reset_index(drop=True) if rows else pd.DataFrame()


# ------------------------- v0.9.0 · selección multipersona -------------------------
def _person_points(person):
    pts = person.get("pose_keypoints_2d", []) if isinstance(person, dict) else []
    if len(pts) < 26 * 3:
        return None
    return np.asarray(pts[:26*3], dtype=float).reshape(26, 3)


def _descriptor_from_points(pts, score_thr=0.20):
    if pts is None or pts.shape[0] < 26:
        return None
    valid = np.isfinite(pts[:,0]) & np.isfinite(pts[:,1]) & np.isfinite(pts[:,2]) & (pts[:,2] >= score_thr)
    if int(valid.sum()) < 7:
        return None

    xy = pts[:, :2]
    # Centro robusto: pelvis/hips/shoulders si son visibles; si no, mediana global.
    core_idx = [HALPE26["Hip"], HALPE26["LHip"], HALPE26["RHip"], HALPE26["LShoulder"], HALPE26["RShoulder"]]
    core_valid = [i for i in core_idx if valid[i]]
    if core_valid:
        center = np.nanmedian(xy[core_valid], axis=0)
    else:
        center = np.nanmedian(xy[valid], axis=0)

    xv, yv = xy[valid,0], xy[valid,1]
    x1, x2 = float(np.nanmin(xv)), float(np.nanmax(xv))
    y1, y2 = float(np.nanmin(yv)), float(np.nanmax(yv))
    w, h = max(1.0, x2-x1), max(1.0, y2-y1)
    scale = float(max(h, w, np.hypot(w, h) * 0.70))
    shape = np.full((26,2), np.nan, dtype=float)
    shape[valid] = (xy[valid] - center) / max(scale, 1.0)

    return {
        "center": np.asarray(center, dtype=float),
        "bbox": np.asarray([x1,y1,x2,y2], dtype=float),
        "scale": scale,
        "shape": shape,
        "valid": valid,
        "mean_score": float(np.nanmean(pts[valid,2])),
        "n_valid": int(valid.sum()),
    }


def _bbox_iou(a, b):
    if a is None or b is None:
        return 0.0
    x1=max(float(a[0]),float(b[0])); y1=max(float(a[1]),float(b[1]))
    x2=min(float(a[2]),float(b[2])); y2=min(float(a[3]),float(b[3]))
    iw=max(0.0,x2-x1); ih=max(0.0,y2-y1)
    inter=iw*ih
    aa=max(0.0,float(a[2]-a[0]))*max(0.0,float(a[3]-a[1]))
    bb=max(0.0,float(b[2]-b[0]))*max(0.0,float(b[3]-b[1]))
    den=aa+bb-inter
    return float(inter/den) if den>0 else 0.0


def _shape_distance(a, b):
    va = a.get("shape"); vb = b.get("shape")
    if va is None or vb is None:
        return 1.0
    ok = np.isfinite(va[:,0]) & np.isfinite(va[:,1]) & np.isfinite(vb[:,0]) & np.isfinite(vb[:,1])
    if int(ok.sum()) < 5:
        return 1.0
    d = np.linalg.norm(va[ok]-vb[ok], axis=1)
    return float(np.nanmedian(d))


def _identity_cost(prev, cand, predicted_center=None):
    if prev is None or cand is None:
        return np.inf
    pc = np.asarray(predicted_center if predicted_center is not None else prev["center"], dtype=float)
    scale = max(float(prev.get("scale",1.0)), float(cand.get("scale",1.0)), 1.0)
    center_d = float(np.linalg.norm(np.asarray(cand["center"])-pc) / scale)
    iou_pen = 1.0 - _bbox_iou(prev.get("bbox"), cand.get("bbox"))
    scale_pen = abs(float(np.log(max(cand.get("scale",1.0),1.0) / max(prev.get("scale",1.0),1.0))))
    shape_pen = min(2.0, _shape_distance(prev, cand))
    conf_pen = max(0.0, 0.55 - float(cand.get("mean_score",0.0)))
    return float(0.42*center_d + 0.23*iou_pen + 0.14*scale_pen + 0.17*shape_pen + 0.04*conf_pen)


def _frame_people(path):
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception:
        return []
    out=[]
    for idx, person in enumerate(data.get("people", []) or []):
        pts=_person_points(person)
        desc=_descriptor_from_points(pts)
        if desc is None:
            continue
        out.append({"person_index":int(idx), "pts":pts, "desc":desc})
    return out


def scan_subject_candidates(json_dir):
    """
    Escoge automáticamente UN FOTOGRAMA de selección con la mayor cantidad de
    sujetos suficientemente visibles. No elige al paciente: solo prepara las
    opciones para que el clínico lo seleccione manualmente.
    """
    paths=sorted(Path(json_dir).glob("*.json"))
    if not paths:
        return None

    best=None
    # Preferir la primera mitad del ensayo para que la elección se haga antes
    # de cruces/giro, pero permitir cualquier frame si allí hay mejor visibilidad.
    scored=[]
    for i,p in enumerate(paths):
        people=_frame_people(p)
        if not people:
            continue
        n=len(people)
        quality=float(np.mean([x["desc"]["mean_score"] for x in people]))
        central_bonus=0.05 if i <= max(1, int(0.55*len(paths))) else 0.0
        score=n*10.0 + quality + central_bonus
        scored.append((score, -i, p, people))
    if not scored:
        return None
    scored.sort(key=lambda x:(x[0],x[1]), reverse=True)
    _,_,p,people=scored[0]
    frame_no=parse_frame_number(p,0)

    # Etiquetas estables para la UI: de izquierda a derecha en el frame elegido.
    ordered=sorted(people, key=lambda x: float(x["desc"]["center"][0]))
    candidates=[]
    for pos, item in enumerate(ordered, start=1):
        d=item["desc"]
        candidates.append({
            "label":f"Sujeto {pos}",
            "person_index":int(item["person_index"]),
            "center_x":float(d["center"][0]),
            "center_y":float(d["center"][1]),
            "bbox":[float(v) for v in d["bbox"]],
            "mean_score":float(d["mean_score"]),
            "n_valid":int(d["n_valid"]),
        })
    return {"frame":int(frame_no), "candidates":candidates, "max_people":int(len(candidates))}


def render_subject_preview(video_path, selection):
    if not selection:
        return None
    cap=cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None
    frame_no=int(selection["frame"])
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no)
    ok, frame=cap.read()
    cap.release()
    if not ok or frame is None:
        return None

    for cand in selection["candidates"]:
        x1,y1,x2,y2=[int(round(v)) for v in cand["bbox"]]
        x1=max(0,x1); y1=max(0,y1); x2=min(frame.shape[1]-1,x2); y2=min(frame.shape[0]-1,y2)
        # Sin colores clínicamente semánticos: etiquetas y cajas de alto contraste.
        cv2.rectangle(frame,(x1,y1),(x2,y2),(255,255,255),3)
        cv2.putText(frame,cand["label"],(x1,max(24,y1-10)),cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,255,255),3,cv2.LINE_AA)
        cv2.putText(frame,cand["label"],(x1,max(24,y1-10)),cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,0,0),1,cv2.LINE_AA)
    ok, enc=cv2.imencode(".jpg",frame,[int(cv2.IMWRITE_JPEG_QUALITY),90])
    return enc.tobytes() if ok else None


def _row_from_tracked(frame_no, pts):
    row={"frame":int(frame_no)}
    for name,idx in HALPE26.items():
        row[f"{name}_x"]=float(pts[idx,0])
        row[f"{name}_y"]=float(pts[idx,1])
        row[f"{name}_score"]=float(pts[idx,2])
    return row



def _identity_anchor_guard(anchor_desc, prev_desc, cand_desc, gap=0):
    """
    v0.10.3 · Barrera adicional contra saltos paciente→acompañante.

    Devuelve (ok, reason). No identifica a la persona por apariencia; solo impide
    cambios geométricos bruscos incompatibles con una continuidad física razonable.
    """
    if anchor_desc is None or prev_desc is None or cand_desc is None:
        return False, "descriptor ausente"

    prev_scale = max(float(prev_desc.get("scale", 1.0)), 1.0)
    cand_scale = max(float(cand_desc.get("scale", 1.0)), 1.0)
    anchor_scale = max(float(anchor_desc.get("scale", 1.0)), 1.0)

    # Cambio brusco frame-a-frame: especialmente útil cuando el terapeuta
    # pasa muy cerca de cámara y aparece súbitamente mucho mayor.
    ratio_prev = cand_scale / prev_scale
    lo = 0.58 if gap <= 2 else 0.48
    hi = 1.72 if gap <= 2 else 1.95
    if not (lo <= ratio_prev <= hi):
        return False, "salto brusco de escala"

    center_jump = float(
        np.linalg.norm(np.asarray(cand_desc["center"]) - np.asarray(prev_desc["center"]))
        / max(prev_scale, cand_scale, 1.0)
    )
    max_jump = 0.78 if gap == 0 else min(1.15, 0.78 + 0.05 * min(gap, 7))
    if center_jump > max_jump:
        return False, "salto espacial excesivo"

    # La forma normalizada no necesita ser idéntica, pero un cambio extremo
    # frente al ancla seleccionada se trata como posible intercambio.
    anchor_shape = _shape_distance(anchor_desc, cand_desc)
    if np.isfinite(anchor_shape) and anchor_shape > 0.95:
        return False, "geometría incompatible con sujeto ancla"

    # Cambio de escala respecto al ancla puede ser grande por perspectiva,
    # pero no arbitrario. Este límite es deliberadamente amplio.
    ratio_anchor = cand_scale / anchor_scale
    if ratio_anchor < 0.28 or ratio_anchor > 3.8:
        return False, "escala incompatible con sujeto ancla"

    return True, "ok"


def _crop_frame_to_subject(frame, row, pad=0.38):
    """
    Recorta visualmente alrededor del sujeto TRACKING seleccionado.
    Solo afecta a la presentación de pestaña 9; no cambia las métricas.
    """
    if frame is None or row is None:
        return frame
    xs, ys = [], []
    for name in HALPE26:
        xk, yk, sk = f"{name}_x", f"{name}_y", f"{name}_score"
        try:
            if xk in row and yk in row and sk in row and float(row[sk]) >= 0.15:
                x, y = float(row[xk]), float(row[yk])
                if np.isfinite(x) and np.isfinite(y):
                    xs.append(x); ys.append(y)
        except Exception:
            continue
    if len(xs) < 5:
        return frame

    h, w = frame.shape[:2]
    x1, x2 = min(xs), max(xs)
    y1, y2 = min(ys), max(ys)
    bw, bh = max(40.0, x2-x1), max(80.0, y2-y1)
    px, py = bw*pad, bh*pad
    xa = max(0, int(round(x1-px)))
    xb = min(w, int(round(x2+px)))
    ya = max(0, int(round(y1-py)))
    yb = min(h, int(round(y2+py)))
    if xb-xa < 80 or yb-ya < 120:
        return frame
    return frame[ya:yb, xa:xb].copy()


def _draw_selected_tracking(frame, frame_no, seg, left_cycle=None, right_cycle=None, crop_subject=True):
    """
    Dibuja SOLO el tracking presente en `seg`, que en modo multipersona es el
    DataFrame del sujeto manualmente bloqueado. Nunca usa personas del vídeo anotado.
    """
    if frame is None:
        return None

    row = None
    if _valid_dataframe(seg):
        match = seg.iloc[(seg["frame"]-int(frame_no)).abs().argsort()[:1]]
        if not match.empty:
            row = match.iloc[0]

    # Dibujar sobre frame limpio.
    if row is not None:
        pairs = [
            ("LShoulder","RShoulder"),("LShoulder","LHip"),("RShoulder","RHip"),
            ("LHip","RHip"),("LHip","LKnee"),("LKnee","LAnkle"),
            ("RHip","RKnee"),("RKnee","RAnkle"),
            ("LAnkle","LHeel"),("LAnkle","LBigToe"),
            ("RAnkle","RHeel"),("RAnkle","RBigToe"),
        ]
        pts = {}
        for name in HALPE26:
            xk,yk,sk=f"{name}_x",f"{name}_y",f"{name}_score"
            try:
                if xk in row and yk in row and sk in row and float(row[sk]) >= 0.15:
                    x,y=float(row[xk]),float(row[yk])
                    if np.isfinite(x) and np.isfinite(y):
                        pts[name]=(int(round(x)),int(round(y)))
            except Exception:
                continue
        for a,b in pairs:
            if a in pts and b in pts:
                cv2.line(frame,pts[a],pts[b],(40,220,80),3,cv2.LINE_AA)
        for p in pts.values():
            cv2.circle(frame,p,4,(0,190,255),-1,cv2.LINE_AA)

    # Fases antes del crop para conservar texto si no se recorta.
    phase_lines=[]
    for cyc,label in [(left_cycle,"IZQ"),(right_cycle,"DER")]:
        info=_cycle_phase_for_frame(cyc,int(frame_no)) if cyc else None
        if info:
            pct,phase=info
            phase_lines.append(f"{label}: {phase} · {pct:.1f}%")

    # Recortar después de dibujar tracking; el texto se añade al recorte.
    if crop_subject and row is not None:
        frame = _crop_frame_to_subject(frame, row, pad=0.42)

    y=30
    for line in phase_lines:
        cv2.putText(frame,line,(12,y),cv2.FONT_HERSHEY_SIMPLEX,0.62,(0,0,0),4,cv2.LINE_AA)
        cv2.putText(frame,line,(12,y),cv2.FONT_HERSHEY_SIMPLEX,0.62,(255,255,255),2,cv2.LINE_AA)
        y += 27

    ok, enc=cv2.imencode(".jpg",frame,[int(cv2.IMWRITE_JPEG_QUALITY),90])
    return enc.tobytes() if ok else None


def load_pose_dataframe_tracked(json_dir, anchor_frame, selected_person_index, prior_boxes=None):
    """
    Seguimiento de identidad bloqueado desde el sujeto elegido manualmente.

    - Sigue hacia delante y hacia atrás desde el frame de selección.
    - Combina continuidad espacial, tamaño corporal, IoU y forma esquelética.
    - Si dos candidatos son demasiado parecidos o el coste es excesivo,
      NO cambia silenciosamente de persona: excluye ese frame.
    """
    paths=sorted(Path(json_dir).glob("*.json"))
    if not paths:
        return pd.DataFrame(), {}

    frames=[]
    for i,p in enumerate(paths):
        fn=parse_frame_number(p,i)
        frames.append((int(fn),p))
    frame_to_pos={fn:i for i,(fn,_) in enumerate(frames)}
    if int(anchor_frame) not in frame_to_pos:
        # frame más cercano, por robustez ante nombres de archivo no contiguos
        anchor_frame=min(frame_to_pos, key=lambda x:abs(x-int(anchor_frame)))
    a_pos=frame_to_pos[int(anchor_frame)]

    anchor_people=_frame_people(frames[a_pos][1])
    anchor=next((x for x in anchor_people if int(x["person_index"])==int(selected_person_index)),None)
    if anchor is None:
        return pd.DataFrame(), {"quality":"No fiable","reason":"No se encontró el sujeto seleccionado en el frame ancla."}

    accepted={int(anchor_frame):anchor}
    ambiguous=set()
    missing=set()
    switches_prevented=0
    max_people=0

    # V72 · Prior espacial procedente de la preselección manual. Se interpola
    # únicamente entre cajas fiables del sujeto elegido y se usa como barrera
    # adicional contra saltos de identidad durante cruces.
    prior_boxes = sorted(prior_boxes or [], key=lambda z:z[0])
    def expected_prior(fn):
        if not prior_boxes:
            return None
        fr=np.asarray([b[0] for b in prior_boxes],float)
        cx=np.asarray([(b[1]+b[3])/2 for b in prior_boxes],float)
        cy=np.asarray([(b[2]+b[4])/2 for b in prior_boxes],float)
        bw=np.asarray([max(1.0,b[3]-b[1]) for b in prior_boxes],float)
        bh=np.asarray([max(1.0,b[4]-b[2]) for b in prior_boxes],float)
        return {
            "center":np.asarray([np.interp(fn,fr,cx,left=cx[0],right=cx[-1]), np.interp(fn,fr,cy,left=cy[0],right=cy[-1])],float),
            "w":float(np.interp(fn,fr,bw,left=bw[0],right=bw[-1])),
            "h":float(np.interp(fn,fr,bh,left=bh[0],right=bh[-1])),
        }

    def walk(indices):
        nonlocal switches_prevented,max_people
        prev=anchor
        prev2=None
        gap=0
        for pos in indices:
            fn,p=frames[pos]
            people=_frame_people(p)
            max_people=max(max_people,len(people))
            if not people:
                missing.add(fn); gap+=1
                continue

            pred=None
            if prev2 is not None and gap==0:
                v=np.asarray(prev["desc"]["center"])-np.asarray(prev2["desc"]["center"])
                pred=np.asarray(prev["desc"]["center"])+v

            scored=[]
            prior=expected_prior(fn)
            for cand in people:
                cost=_identity_cost(prev["desc"],cand["desc"],predicted_center=pred)
                prior_center=np.nan; prior_scale=np.nan
                if prior is not None:
                    c=np.asarray(cand["desc"]["center"],float)
                    pscale=max(prior["w"],prior["h"],1.0)
                    prior_center=float(np.linalg.norm(c-prior["center"])/pscale)
                    cscale=max(float(cand["desc"].get("scale",1.0)),1.0)
                    prior_scale=abs(float(np.log(cscale/pscale)))
                    # El prior manual pesa de forma fuerte durante un cruce: un
                    # candidato lejos de la trayectoria esperada no puede ganar
                    # solo por proximidad al candidato del frame anterior.
                    cost += 0.60*prior_center + 0.16*prior_scale
                scored.append((cost,cand,prior_center,prior_scale))
            scored.sort(key=lambda x:x[0])
            best_cost,best,best_prior_center,best_prior_scale=scored[0]
            second_cost=scored[1][0] if len(scored)>1 else np.inf
            margin=second_cost-best_cost

            # Umbral algo más permisivo tras una oclusión breve, pero nunca
            # permite un salto grande de identidad.
            max_cost=0.92 if gap==0 else min(1.15,0.92+0.025*min(gap,9))
            guard_ok, guard_reason = _identity_anchor_guard(
                anchor["desc"], prev["desc"], best["desc"], gap=gap
            )
            # V73 · el simple hecho de que dos personas estén próximas NO invalida
            # automáticamente al paciente. En V71/V72 esto generaba un efecto cascada:
            # al aparecer el acompañante se descartaba un frame, aumentaba ``gap`` y la
            # reacquisición se volvía cada vez más difícil hasta perder al paciente.
            # Ahora solo se declara cruce peligroso cuando el solapamiento/proximidad es
            # realmente alto Y la identidad del mejor candidato no está bien anclada.
            crossing_risk = False
            iou12 = 0.0
            cdist12 = np.inf
            if len(scored) > 1:
                second = scored[1][1]
                d1, d2 = best["desc"], second["desc"]
                iou12 = _bbox_iou(d1.get("bbox"), d2.get("bbox"))
                cdist12 = float(
                    np.linalg.norm(np.asarray(d1["center"]) - np.asarray(d2["center"]))
                    / max(min(float(d1.get("scale",1.0)), float(d2.get("scale",1.0))), 1.0)
                )
                crossing_risk = bool(iou12 >= 0.22 or cdist12 <= 0.42)

            anchor_shape = _shape_distance(anchor["desc"], best["desc"])

            # Identidad fuertemente compatible con el paciente elegido. Si se cumple,
            # se mantiene el seguimiento aunque el acompañante esté cerca o parcialmente
            # solapado. Esto permite seguir al Sujeto 1 mientras sigue siendo visible.
            strong_identity = bool(
                best_cost <= 0.66 and
                (not np.isfinite(best_prior_center) or best_prior_center <= 0.55) and
                (not np.isfinite(anchor_shape) or anchor_shape <= 0.70) and
                (len(scored) == 1 or margin >= 0.08)
            )

            # Reentrada tras una oclusión: basta con una coincidencia sólida con el prior
            # espacial y la firma corporal; no se exige una separación extrema respecto
            # al segundo sujeto. Así se evita que ``gap`` bloquee para siempre la pista.
            strong_reacq = bool(
                gap > 0 and
                best_cost <= 0.82 and
                (not np.isfinite(best_prior_center) or best_prior_center <= 0.62) and
                (not np.isfinite(anchor_shape) or anchor_shape <= 0.78) and
                (len(scored) == 1 or margin >= 0.07)
            )

            reacq_weak = bool(
                gap > 0 and not strong_reacq and
                (margin < 0.12 or best_cost > 0.92)
            )
            anchor_weak = bool(
                np.isfinite(anchor_shape) and
                anchor_shape > (0.98 if len(scored)>1 else 1.12)
            )

            prior_weak = bool(
                np.isfinite(best_prior_center) and
                (best_prior_center > (0.92 if len(scored)>1 else 1.12) or
                 (np.isfinite(best_prior_scale) and best_prior_scale > 0.92))
            )

            ambiguous_match = (
                (prior_weak and not strong_identity and not strong_reacq) or
                (best_cost > max_cost and not strong_reacq) or
                (len(scored)>1 and margin < 0.08 and best_cost > 0.28 and not strong_identity) or
                (crossing_risk and len(scored)>1 and margin < 0.20 and not strong_identity and not strong_reacq) or
                reacq_weak or
                (anchor_weak and not strong_reacq) or
                ((not guard_ok) and not strong_reacq)
            )
            if ambiguous_match:
                ambiguous.add(fn)
                switches_prevented += 1 if len(scored)>1 else 0
                gap += 1
                # Nunca adoptar un candidato durante un cruce u oclusión dudosa.
                continue

            accepted[fn]=best
            prev2=prev
            prev=best
            gap=0

    walk(range(a_pos+1,len(frames)))
    # Reiniciar referencia para seguimiento hacia atrás.
    prev_anchor=anchor
    # La función walk mantiene variables locales prev, por lo que basta otra llamada
    walk(range(a_pos-1,-1,-1))

    # V71 · cuarentena temporal alrededor de los frames ambiguos. Un cambio de
    # identidad puede empezar unos instantes antes del máximo solapamiento y
    # persistir unos frames después. Eliminamos un halo corto alrededor de cada
    # cruce para que esos datos no lleguen a métricas, eventos ni gráficas.
    quarantine=set()
    if ambiguous:
        frame_numbers=[fn for fn,_ in frames]
        frame_set=set(frame_numbers)
        halo=2  # V73: ~0.07 s a 30 fps; evita contaminar el cruce sin borrar tramos prolongados
        for af in ambiguous:
            for qf in range(int(af)-halo, int(af)+halo+1):
                if qf in frame_set:
                    quarantine.add(qf)
        for qf in quarantine:
            accepted.pop(qf, None)

    rows=[]
    for fn,_ in frames:
        item=accepted.get(fn)
        if item is not None:
            rows.append(_row_from_tracked(fn,item["pts"]))

    df=pd.DataFrame(rows).sort_values("frame").reset_index(drop=True) if rows else pd.DataFrame()
    total=len(frames)
    reliable=len(rows)
    excluded=max(0,total-reliable)
    continuity=100.0*reliable/total if total else np.nan
    ambiguous_pct=100.0*len(ambiguous)/total if total else np.nan
    missing_pct=100.0*len(missing)/total if total else np.nan

    if np.isfinite(continuity):
        quality="Alta" if continuity>=90 else ("Moderada" if continuity>=75 else "Baja")
    else:
        quality="No fiable"
    info={
        "manual":True,
        "anchor_frame":int(anchor_frame),
        "selected_person_index":int(selected_person_index),
        "identity_continuity_pct":float(continuity) if np.isfinite(continuity) else np.nan,
        "ambiguous_excluded_pct":float(ambiguous_pct) if np.isfinite(ambiguous_pct) else np.nan,
        "missing_pose_pct":float(missing_pct) if np.isfinite(missing_pct) else np.nan,
        "frames_total":int(total),
        "frames_reliable":int(reliable),
        "frames_excluded":int(excluded),
        "ambiguous_frames":int(len(ambiguous)),
        "quarantine_frames":int(len(quarantine)),
        "switches_prevented":int(switches_prevented),
        "max_people_detected":int(max_people),
        "quality":quality,
    }
    return df,info


def tracking_metrics(info, prefix="", view_label=""):
    if not info or not info.get("manual"):
        return []
    p=(prefix+"_" if prefix else "")
    view_note=f" ({view_label})" if view_label else ""
    return [
        {"key":p+"subject_manual_selection_flag","label":"Selección manual de sujeto"+view_note,"value":1.0,"unit":"bool","quality":"Bloqueado","notes":"v0.9.1: el paciente fue seleccionado explícitamente por el clínico antes del análisis biomecánico."},
        {"key":p+"identity_continuity_pct","label":"Continuidad de identidad"+view_note,"value":info.get("identity_continuity_pct"),"unit":"%","quality":info.get("quality","No fiable"),"notes":"Porcentaje de frames en los que la identidad seleccionada se mantuvo con confianza suficiente."},
        {"key":p+"identity_ambiguous_excluded_pct","label":"Frames ambiguos excluidos"+view_note,"value":info.get("ambiguous_excluded_pct"),"unit":"%","quality":"Control interno","notes":"Frames descartados por riesgo de confundir al paciente con otra persona."},
        {"key":p+"identity_frames_excluded","label":"Frames excluidos por identidad"+view_note,"value":info.get("frames_excluded"),"unit":"frames","quality":"Control interno","notes":"Incluye oclusiones, pérdidas de pose y emparejamientos ambiguos; nunca se sustituyen silenciosamente por otro sujeto."},
        {"key":p+"identity_quarantine_frames","label":"Frames en cuarentena por cruce/oclusión"+view_note,"value":info.get("quarantine_frames",0),"unit":"frames","quality":"Control interno","notes":"V72: frames ambiguos y un margen temporal alrededor del cruce se excluyen de métricas y gráficas; la reacquisición usa además el prior espacial del sujeto elegido."},
        {"key":p+"max_people_detected","label":"Máximo de sujetos detectados"+view_note,"value":info.get("max_people_detected"),"unit":"personas","quality":"Directa","notes":"Máximo número de personas con pose utilizable observado durante el seguimiento."},
    ]


def point_angle(ax, ay, bx, by, cx, cy):
    ba = np.array([ax - bx, ay - by], dtype=float)
    bc = np.array([cx - bx, cy - by], dtype=float)
    nba, nbc = np.linalg.norm(ba), np.linalg.norm(bc)
    if nba == 0 or nbc == 0:
        return np.nan
    c = np.clip(np.dot(ba, bc) / (nba * nbc), -1.0, 1.0)
    return float(np.degrees(np.arccos(c)))


def robust_rom(x):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    return float(np.percentile(x, 95) - np.percentile(x, 5)) if len(x) >= 5 else np.nan


def rolling_smooth(arr, window=7):
    return pd.Series(arr).rolling(window, center=True, min_periods=1).mean().to_numpy()


def zero_crossings(signal):
    s = np.asarray(signal, dtype=float)
    out = []
    for i in range(1, len(s)):
        if not (np.isfinite(s[i-1]) and np.isfinite(s[i])):
            continue
        if (s[i-1] <= 0 < s[i]) or (s[i-1] >= 0 > s[i]):
            out.append(i)
    return np.asarray(out, dtype=int)


def quality_label(score):
    return "Alta" if score >= 0.80 else ("Moderada" if score >= 0.65 else "Baja")


def add_angle_columns(seg):
    seg = seg.copy()
    for side in ("L", "R"):
        knee, hip, ankle, shoulder = [], [], [], []
        for _, r in seg.iterrows():
            ka = point_angle(r[f"{side}Hip_x"], r[f"{side}Hip_y"], r[f"{side}Knee_x"], r[f"{side}Knee_y"], r[f"{side}Ankle_x"], r[f"{side}Ankle_y"])
            ha = point_angle(r[f"{side}Shoulder_x"], r[f"{side}Shoulder_y"], r[f"{side}Hip_x"], r[f"{side}Hip_y"], r[f"{side}Knee_x"], r[f"{side}Knee_y"])
            aa = point_angle(r[f"{side}Knee_x"], r[f"{side}Knee_y"], r[f"{side}Ankle_x"], r[f"{side}Ankle_y"], r[f"{side}BigToe_x"], r[f"{side}BigToe_y"])
            sa = point_angle(r[f"{side}Hip_x"], r[f"{side}Hip_y"], r[f"{side}Shoulder_x"], r[f"{side}Shoulder_y"], r[f"{side}Elbow_x"], r[f"{side}Elbow_y"])
            knee.append(180.0 - ka if np.isfinite(ka) else np.nan)
            hip.append(180.0 - ha if np.isfinite(ha) else np.nan)
            ankle.append(aa)
            shoulder.append(sa)
        seg[f"{side}_knee_flex"] = knee
        seg[f"{side}_hip_flex"] = hip
        seg[f"{side}_ankle_angle"] = ankle
        seg[f"{side}_shoulder_elev"] = shoulder
    return seg


def axis_angle_to_vertical(x1, y1, x2, y2):
    """Ángulo firmado de un eje 2D respecto a la vertical de la imagen, plegado a [-90, 90]."""
    dx, dy = float(x2-x1), float(y2-y1)
    if not np.isfinite(dx) or not np.isfinite(dy) or (abs(dx)+abs(dy) == 0):
        return np.nan
    a = float(np.degrees(np.arctan2(dx, -dy)))
    while a > 90: a -= 180
    while a < -90: a += 180
    return a


def axis_angle_to_horizontal(x1, y1, x2, y2):
    dx, dy = float(x2-x1), float(y2-y1)
    if not np.isfinite(dx) or not np.isfinite(dy) or (abs(dx)+abs(dy) == 0):
        return np.nan
    a = float(np.degrees(np.arctan2(dy, dx)))
    while a > 90: a -= 180
    while a < -90: a += 180
    return a


def add_frontal_columns(seg):
    """Métricas proyectadas para vista frontal/posterior. No equivalen a rotaciones 3D ni a pronación clínica."""
    seg = seg.copy()
    for side in ("L", "R"):
        knee_dev, foot_prog, rearfoot = [], [], []
        for _, r in seg.iterrows():
            ka = point_angle(r[f"{side}Hip_x"], r[f"{side}Hip_y"], r[f"{side}Knee_x"], r[f"{side}Knee_y"], r[f"{side}Ankle_x"], r[f"{side}Ankle_y"])
            knee_dev.append(abs(180.0-ka) if np.isfinite(ka) else np.nan)
            foot_prog.append(axis_angle_to_vertical(r[f"{side}Heel_x"], r[f"{side}Heel_y"], r[f"{side}BigToe_x"], r[f"{side}BigToe_y"]))
            rearfoot.append(axis_angle_to_vertical(r[f"{side}Ankle_x"], r[f"{side}Ankle_y"], r[f"{side}Heel_x"], r[f"{side}Heel_y"]))
        seg[f"{side}_frontal_knee_dev"] = knee_dev
        seg[f"{side}_foot_progress_proj"] = foot_prog
        seg[f"{side}_rearfoot_tilt_proj"] = rearfoot
    seg["pelvis_obliquity"] = [axis_angle_to_horizontal(r.LHip_x, r.LHip_y, r.RHip_x, r.RHip_y) for _, r in seg.iterrows()]
    seg["shoulder_obliquity"] = [axis_angle_to_horizontal(r.LShoulder_x, r.LShoulder_y, r.RShoulder_x, r.RShoulder_y) for _, r in seg.iterrows()]
    # Tronco en plano frontal/posterior: eje entre el centro pélvico y el centro de hombros.
    trunk_lean = []
    for _, r in seg.iterrows():
        hip_x = (r.LHip_x + r.RHip_x) / 2.0
        hip_y = (r.LHip_y + r.RHip_y) / 2.0
        sh_x = (r.LShoulder_x + r.RShoulder_x) / 2.0
        sh_y = (r.LShoulder_y + r.RShoulder_y) / 2.0
        trunk_lean.append(axis_angle_to_vertical(hip_x, hip_y, sh_x, sh_y))
    seg["trunk_lateral_lean"] = trunk_lean
    seg["shoulder_pelvis_rel"] = seg["shoulder_obliquity"] - seg["pelvis_obliquity"]
    pelvis_w = np.abs(seg.RHip_x.to_numpy(float) - seg.LHip_x.to_numpy(float))
    ankle_w = np.abs(seg.RAnkle_x.to_numpy(float) - seg.LAnkle_x.to_numpy(float))
    seg["base_width_relative"] = np.divide(ankle_w, pelvis_w, out=np.full_like(ankle_w, np.nan), where=pelvis_w>1e-6)
    return seg


def _foot_centroid(seg, side):
    x = (seg[f"{side}Ankle_x"].to_numpy(float) + seg[f"{side}Heel_x"].to_numpy(float) + seg[f"{side}BigToe_x"].to_numpy(float)) / 3.0
    y = (seg[f"{side}Ankle_y"].to_numpy(float) + seg[f"{side}Heel_y"].to_numpy(float) + seg[f"{side}BigToe_y"].to_numpy(float)) / 3.0
    return x, y


def _bool_runs(mask):
    """Devuelve runs True como (inicio, fin_exclusivo)."""
    m = np.asarray(mask, dtype=bool)
    runs = []
    start = None
    for i, v in enumerate(m):
        if v and start is None:
            start = i
        if start is not None and ((not v) or i == len(m) - 1):
            end = i if not v else i + 1
            runs.append((int(start), int(end)))
            start = None
    return runs


def _fill_short_false_gaps(mask, max_gap_frames, speed=None, speed_ceiling=None):
    m = np.asarray(mask, dtype=bool).copy()
    if len(m) == 0 or max_gap_frames <= 0:
        return m
    for a, b in _bool_runs(~m):
        if a == 0 or b == len(m):
            continue
        if (b - a) <= max_gap_frames:
            if speed is None or speed_ceiling is None:
                m[a:b] = True
            else:
                g = np.asarray(speed[a:b], float)
                if len(g) and np.isfinite(g).any() and float(np.nanmedian(g)) <= float(speed_ceiling):
                    m[a:b] = True
    return m


def _remove_short_true_runs(mask, min_run_frames):
    m = np.asarray(mask, dtype=bool).copy()
    for a, b in _bool_runs(m):
        if (b - a) < min_run_frames:
            m[a:b] = False
    return m


def _support_mask_2d(seg, side, fps, expected_stride_s=np.nan):
    """
    v0.9.2: estimación cinemática 2D de apoyo con continuidad temporal.
    Evita fragmentar un apoyo real en microbloques por jitter del tracking.

    Sigue siendo markerless 2D: NO sustituye footswitch, plataforma de fuerzas
    ni presión plantar.
    """
    x, y = _foot_centroid(seg, side)
    xs = (
        pd.Series(x).interpolate(limit_direction="both")
        .rolling(5, center=True, min_periods=1).median()
        .rolling(5, center=True, min_periods=1).mean()
        .to_numpy(float)
    )
    ys = (
        pd.Series(y).interpolate(limit_direction="both")
        .rolling(5, center=True, min_periods=1).median()
        .rolling(5, center=True, min_periods=1).mean()
        .to_numpy(float)
    )
    speed_px_s = np.hypot(np.gradient(xs), np.gradient(ys)) * float(fps)

    hip_x = (seg.LHip_x.to_numpy(float) + seg.RHip_x.to_numpy(float)) / 2.0
    hip_y = (seg.LHip_y.to_numpy(float) + seg.RHip_y.to_numpy(float)) / 2.0
    sh_x = (seg.LShoulder_x.to_numpy(float) + seg.RShoulder_x.to_numpy(float)) / 2.0
    sh_y = (seg.LShoulder_y.to_numpy(float) + seg.RShoulder_y.to_numpy(float)) / 2.0
    torso = np.hypot(sh_x - hip_x, sh_y - hip_y)
    torso = pd.Series(torso).replace([np.inf, -np.inf], np.nan).interpolate(limit_direction="both").to_numpy(float)
    med_torso = float(np.nanmedian(torso)) if np.isfinite(torso).any() else np.nan
    if not np.isfinite(med_torso) or med_torso < 5:
        med_torso = 1.0
    speed = speed_px_s / med_torso

    finite = speed[np.isfinite(speed)]
    if len(finite) < max(20, int(fps)):
        return np.zeros(len(seg), dtype=bool), speed, {
            "quality": "No fiable", "score": 0.0, "reason": "Señal de pie insuficiente."
        }

    low = float(np.nanpercentile(finite, 38))
    high = float(np.nanpercentile(finite, 72))
    if high <= low:
        high = low + max(float(np.nanstd(finite)) * 0.35, 1e-6)

    state_support = bool(speed[0] <= (low + high) / 2.0) if np.isfinite(speed[0]) else True
    mask = np.zeros(len(speed), dtype=bool)
    for i, s in enumerate(speed):
        if not np.isfinite(s):
            mask[i] = state_support
            continue
        if state_support and s > high:
            state_support = False
        elif (not state_support) and s < low:
            state_support = True
        mask[i] = state_support

    if np.isfinite(expected_stride_s) and expected_stride_s > 0:
        gap_s = min(0.40, max(0.12, 0.09 * float(expected_stride_s)))
        min_support_s = min(0.80, max(0.18, 0.14 * float(expected_stride_s)))
        min_swing_s = min(0.50, max(0.12, 0.07 * float(expected_stride_s)))
    else:
        gap_s, min_support_s, min_swing_s = 0.20, 0.20, 0.12

    mask = _fill_short_false_gaps(
        mask,
        max_gap_frames=max(1, int(round(gap_s * fps))),
        speed=speed,
        speed_ceiling=high * 1.10,
    )
    mask = _remove_short_true_runs(mask, max(1, int(round(min_support_s * fps))))

    swing = ~mask
    swing = _remove_short_true_runs(swing, max(1, int(round(min_swing_s * fps))))
    mask = ~swing
    mask = _remove_short_true_runs(mask, max(1, int(round(min_support_s * fps))))

    return mask.astype(bool), speed, {
        "quality": "Pendiente", "score": np.nan,
        "reason": "Máscara estabilizada por histéresis y continuidad temporal.",
        "low_thr": low, "high_thr": high,
    }


def _support_cycle_summary(mask, fps, expected_stride_s=np.nan):
    """
    Convierte una máscara continua en ciclos IC→TO→siguiente IC y puntúa
    su consistencia. Los límites son controles matemáticos amplios, no
    rangos clínicos de normalidad.
    """
    starts, ends = _edges(mask)
    stance, swing, cycles, stance_pct = [], [], [], []
    valid_ic, valid_to, valid_next_ic = [], [], []

    for j in range(len(starts) - 1):
        ic = int(starts[j])
        nxt = int(starts[j + 1])
        offs = ends[(ends > ic) & (ends <= nxt)]
        if not len(offs):
            continue
        to = int(offs[0])
        cyc = (nxt - ic) / float(fps)
        st = (to - ic) / float(fps)
        sw = (nxt - to) / float(fps)
        if cyc <= 0 or st <= 0 or sw < 0:
            continue
        ratio = st / cyc
        if ratio < 0.10 or ratio > 0.985:
            continue
        if np.isfinite(expected_stride_s) and expected_stride_s > 0:
            if cyc < 0.45 * expected_stride_s or cyc > 1.80 * expected_stride_s:
                continue

        stance.append(st)
        swing.append(sw)
        cycles.append(cyc)
        stance_pct.append(ratio * 100.0)
        valid_ic.append(ic)
        valid_to.append(to)
        valid_next_ic.append(nxt)

    stance = np.asarray(stance, float)
    swing = np.asarray(swing, float)
    cycles = np.asarray(cycles, float)
    stance_pct = np.asarray(stance_pct, float)

    score = 100.0
    reasons = []

    if len(cycles) < 2:
        score -= 55
        reasons.append("menos de 2 ciclos completos")
    elif len(cycles) < 3:
        score -= 15
        reasons.append("pocos ciclos completos")

    err = np.nan
    if len(cycles) and np.isfinite(expected_stride_s) and expected_stride_s > 0:
        cyc_med = float(np.nanmedian(cycles))
        err = abs(cyc_med - expected_stride_s) / expected_stride_s * 100.0
        if err > 50:
            score -= 45
            reasons.append("periodo de ciclo discordante con la cadencia")
        elif err > 30:
            score -= 25
            reasons.append("periodo de ciclo moderadamente discordante")
        elif err > 18:
            score -= 10
            reasons.append("periodo de ciclo con ligera discordancia")

    fragmentation = np.nan
    if np.isfinite(expected_stride_s) and expected_stride_s > 0 and len(mask) > 0:
        duration = len(mask) / float(fps)
        expected_same_side_cycles = max(duration / expected_stride_s, 1.0)
        fragmentation = len(starts) / expected_same_side_cycles
        if fragmentation > 2.2:
            score -= 35
            reasons.append("segmentación fragmentada")
        elif fragmentation > 1.6:
            score -= 15
            reasons.append("posible fragmentación residual")

    score = float(max(0.0, min(100.0, score)))
    quality = "Alta" if score >= 75 else ("Moderada" if score >= 55 else "No fiable")

    return {
        "stance": stance,
        "swing": swing,
        "cycle": cycles,
        "stance_pct": stance_pct,
        "swing_pct": 100.0 - stance_pct if len(stance_pct) else np.asarray([], float),
        "ic": np.asarray(valid_ic, int),
        "to": np.asarray(valid_to, int),
        "next_ic": np.asarray(valid_next_ic, int),
        "starts": np.asarray(starts, int),
        "ends": np.asarray(ends, int),
        "quality": quality,
        "score": score,
        "cycle_error_pct": float(err) if np.isfinite(err) else np.nan,
        "fragmentation_index": float(fragmentation) if np.isfinite(fragmentation) else np.nan,
        "reason": ", ".join(reasons) if reasons else "consistencia temporal suficiente",
    }



def _automatic_straight_walking_mask(seg, fps):
    """
    v0.9.2: detecta y excluye automáticamente transiciones/giro de 180° en
    registros frontal/posterior.

    Combina dos señales normalizadas:
      1) estrechamiento transitorio del ancho de hombros/caderas respecto al
         tronco (el cuerpo se pone de perfil durante el giro);
      2) inversión sostenida de la tendencia de tamaño corporal aparente
         (alejarse -> acercarse, o viceversa).

    Es un filtro de calidad, no un detector clínico de giro.
    Si la evidencia no es suficiente, solo recorta bordes del registro.
    """
    n = len(seg)
    if n == 0:
        return np.zeros(0, dtype=bool), {
            "turn_detected": False, "excluded_pct": np.nan,
            "reason": "segmento vacío"
        }

    fps = float(max(fps, 1.0))
    edge = min(max(1, int(round(0.35 * fps))), max(1, n // 8))
    valid = np.ones(n, dtype=bool)
    if n > 2 * edge:
        valid[:edge] = False
        valid[-edge:] = False

    lx = seg.LShoulder_x.to_numpy(float); rx = seg.RShoulder_x.to_numpy(float)
    ly = seg.LShoulder_y.to_numpy(float); ry = seg.RShoulder_y.to_numpy(float)
    lhx = seg.LHip_x.to_numpy(float); rhx = seg.RHip_x.to_numpy(float)
    lhy = seg.LHip_y.to_numpy(float); rhy = seg.RHip_y.to_numpy(float)

    shoulder_w = np.abs(rx - lx)
    hip_w = np.abs(rhx - lhx)
    shoulder_y = (ly + ry) / 2.0
    hip_y = (lhy + rhy) / 2.0
    torso_h = np.abs(hip_y - shoulder_y)

    def _smooth(a, w):
        return (
            pd.Series(a).replace([np.inf, -np.inf], np.nan)
            .interpolate(limit_direction="both")
            .rolling(w, center=True, min_periods=1).median()
            .rolling(w, center=True, min_periods=1).mean()
            .to_numpy(float)
        )

    w = max(5, int(round(0.25 * fps)) | 1)
    shoulder_w = _smooth(shoulder_w, w)
    hip_w = _smooth(hip_w, w)
    torso_h = _smooth(torso_h, w)

    denom = np.maximum(torso_h, 1e-6)
    frontal_ratio = 0.65 * (shoulder_w / denom) + 0.35 * (hip_w / denom)
    finite_ratio = frontal_ratio[np.isfinite(frontal_ratio)]
    turn_candidates = np.zeros(n, dtype=bool)

    # Evidencia 1: orientación transitoria de perfil.
    if len(finite_ratio) >= max(20, int(fps)):
        reference = float(np.nanpercentile(finite_ratio, 75))
        if np.isfinite(reference) and reference > 1e-6:
            turn_candidates |= frontal_ratio < (0.62 * reference)

    # Evidencia 2: inversión sostenida del tamaño aparente del cuerpo.
    # Evita declarar giro por un único frame: compara tendencias a ambos lados.
    scale = _smooth(torso_h, max(7, int(round(0.40 * fps)) | 1))
    look = max(5, int(round(0.85 * fps)))
    reversal_scores = np.zeros(n, dtype=float)
    for i in range(look, n - look):
        pre = scale[i] - scale[i - look]
        post = scale[i + look] - scale[i]
        local = float(np.nanmedian(scale[max(0, i-look):min(n, i+look+1)]))
        if not np.isfinite(local) or local <= 1e-6:
            continue
        # signo opuesto y cambio conjunto de al menos ~5 % del tamaño local
        if np.isfinite(pre) and np.isfinite(post) and pre * post < 0:
            reversal_scores[i] = (abs(pre) + abs(post)) / local

    if np.nanmax(reversal_scores) >= 0.05:
        i0 = int(np.nanargmax(reversal_scores))
        turn_candidates[i0] = True

    # Agrupa el giro y añade margen temporal para no contaminar ciclos vecinos.
    turn_detected = bool(turn_candidates.any())
    if turn_detected:
        idx = np.where(turn_candidates)[0]
        # Si hay muchos candidatos, conservar el cluster principal alrededor
        # del punto con menor frontalidad / mayor inversión.
        center = int(np.median(idx))
        if np.nanmax(reversal_scores) >= 0.05:
            center = int(np.nanargmax(reversal_scores))
        pad = max(1, int(round(0.75 * fps)))
        a = max(0, center - pad)
        b = min(n, center + pad + 1)
        valid[a:b] = False

    # No aplicar una exclusión automática agresiva si deja poco material útil.
    min_keep = max(int(round(3.0 * fps)), int(round(0.45 * n)))
    if valid.sum() < min_keep:
        valid[:] = True
        if n > 2 * edge:
            valid[:edge] = False
            valid[-edge:] = False
        turn_detected = False
        reason = "evidencia de giro no suficientemente robusta; solo se recortan bordes"
    else:
        reason = "giro/transición excluido automáticamente" if turn_detected else "sin giro robusto detectado; bordes excluidos"

    return valid, {
        "turn_detected": turn_detected,
        "excluded_pct": float((1.0 - valid.mean()) * 100.0),
        "usable_pct": float(valid.mean() * 100.0),
        "reason": reason,
    }


def _straight_block_ids(valid_mask):
    """V74 · asigna 1..N a cada bloque rectilíneo continuo y 0 al dominio excluido."""
    m = np.asarray(valid_mask, dtype=bool)
    ids = np.zeros(len(m), dtype=int)
    bid = 0; start = None
    for i, ok in enumerate(m):
        if ok and start is None:
            bid += 1; start = i
        if ok:
            ids[i] = bid
        elif start is not None:
            start = None
    return ids


def _mask_seg_to_straight_domain(seg, valid_mask):
    """
    V74 · máscara maestra para TODA la cadena biomecánica.
    Conserva el eje temporal/frame y los indicadores de dominio, pero convierte
    en NaN cualquier coordenada, score o variable numérica durante giro/bordes.
    Las fórmulas existentes no cambian: reciben exactamente las mismas columnas,
    solo que los frames no rectilíneos ya no contienen datos analizables.
    """
    out = seg.copy()
    m = np.asarray(valid_mask, dtype=bool)
    if len(out) != len(m):
        raise ValueError("Máscara rectilínea incompatible con el segmento.")
    out["straight_walking_valid"] = m
    out["straight_block_id"] = _straight_block_ids(m)
    preserve = {"frame", "straight_walking_valid", "straight_block_id", "L_support_2d", "R_support_2d"}
    bad = ~m
    for c in out.columns:
        if c in preserve:
            continue
        if pd.api.types.is_numeric_dtype(out[c]):
            out.loc[bad, c] = np.nan
    return out


def _kinematic_alternation_metrics_straight_blocks(seg, fps, valid_mask):
    """
    V74 · ejecuta el detector cinemático SIN atravesar el cambio de sentido.
    Se llama a la fórmula existente de forma independiente en cada bloque
    rectilíneo y después se combinan solo intervalos/eventos internos de bloque.
    """
    m = np.asarray(valid_mask, dtype=bool)
    block_ids = _straight_block_ids(m)
    parts = []
    for bid in [x for x in np.unique(block_ids) if x > 0]:
        idx = np.where(block_ids == bid)[0]
        if len(idx) < max(45, int(round(2.5*fps))):
            continue
        sub = seg.iloc[idx].copy().reset_index(drop=True)
        r = _kinematic_alternation_metrics(sub, fps)
        if int(r.get("n_intervals",0) or 0) <= 0:
            continue
        # Convertir posiciones locales a frames reales del bloque.
        evf = np.asarray(r.get("event_frames", []), int)
        if len(evf) == 0 and len(r.get("events", [])):
            pos = np.asarray(r.get("events", []), int)
            pos = pos[(pos >= 0) & (pos < len(sub))]
            evf = sub.frame.to_numpy(int)[pos]
        rr = dict(r); rr["event_frames"] = evf; rr["block_id"] = int(bid)
        parts.append(rr)
    if not parts:
        return _kinematic_alternation_metrics(_mask_seg_to_straight_domain(seg, m), fps)
    intervals = np.concatenate([np.asarray(r.get("intervals", []), float) for r in parts])
    intervals = intervals[np.isfinite(intervals) & (intervals > 0)]
    event_frames = np.concatenate([np.asarray(r.get("event_frames", []), int) for r in parts]) if parts else np.asarray([],int)
    cadence = float(60.0/np.nanmedian(intervals)) if len(intervals) else np.nan
    cv = float(np.nanstd(intervals, ddof=1)/np.nanmean(intervals)*100.0) if len(intervals)>=3 and np.nanmean(intervals)>0 else np.nan
    # Asimetría se conserva como promedio de los bloques en que la fórmula pudo calcularla.
    asym_vals = [float(r.get("asym")) for r in parts if np.isfinite(r.get("asym",np.nan))]
    asym = float(np.nanmean(asym_vals)) if asym_vals else np.nan
    periodic_vals = [float(r.get("periodicity")) for r in parts if np.isfinite(r.get("periodicity",np.nan))]
    periodicity = float(np.nanmean(periodic_vals)) if periodic_vals else np.nan
    nint = int(sum(int(r.get("n_intervals",0) or 0) for r in parts))
    quality = "Alta" if nint >= 6 else ("Moderada" if nint >= 4 else "Baja")
    return {
        "cadence": cadence, "asym": asym, "cv": cv,
        "events": np.asarray([], int), "event_frames": np.sort(event_frames),
        "intervals": intervals, "n_intervals": nint,
        "quality": quality, "periodicity": periodicity,
        "reason": f"V74: {len(parts)} bloque(s) rectilíneo(s) analizados independientemente; giro no interpolado",
        "blocks_used": int(len(parts)),
    }


def _filter_support_summary_to_mask(summary, valid_mask, fps, min_fraction=0.90):
    """
    Conserva únicamente ciclos IC→TO→siguiente IC contenidos casi por completo
    en el dominio rectilíneo válido.
    """
    valid_mask = np.asarray(valid_mask, dtype=bool)
    keys = ["stance", "swing", "cycle", "stance_pct", "swing_pct", "ic", "to", "next_ic"]
    arrs = {k: np.asarray(summary.get(k, [])) for k in keys}

    ncyc = len(arrs["ic"])
    keep = []
    for j in range(ncyc):
        ic = int(arrs["ic"][j])
        nxt = int(arrs["next_ic"][j]) if j < len(arrs["next_ic"]) else int(round(ic + arrs["cycle"][j] * fps))
        a, b = max(0, ic), min(len(valid_mask), max(ic + 1, nxt))
        frac = float(valid_mask[a:b].mean()) if b > a else 0.0
        keep.append(frac >= float(min_fraction))
    keep = np.asarray(keep, dtype=bool)

    out = dict(summary)
    for k in keys:
        if len(arrs[k]) == ncyc:
            out[k] = arrs[k][keep]

    # Cobertura temporal de ciclos aceptados.
    coverage = np.zeros(len(valid_mask), dtype=bool)
    for ic, nxt in zip(np.asarray(out.get("ic", []), int), np.asarray(out.get("next_ic", []), int)):
        coverage[max(0, ic):min(len(coverage), max(ic + 1, nxt))] = True
    out["coverage_mask"] = coverage & valid_mask

    # Reevalúa la calidad si la exclusión deja muy pocos ciclos.
    n = len(out.get("cycle", []))
    score = float(summary.get("score", 0.0))
    reasons = [str(summary.get("reason", ""))]
    if n < 2:
        score = min(score, 45.0)
        reasons.append("menos de 2 ciclos rectilíneos válidos")
    elif n < 3:
        score = min(score, 70.0)
        reasons.append("solo 2 ciclos rectilíneos válidos")
    quality = "Alta" if score >= 75 else ("Moderada" if score >= 55 else "No fiable")
    out["score"] = score
    out["quality"] = quality
    out["reason"] = ", ".join([r for r in reasons if r])
    return out


def _contact_step_metrics(L, R, fps):
    """
    Cadencia, CV y asimetría a partir de contactos iniciales I/D de ciclos
    validados. Evita usar cruces geométricos pares/impares como sustituto de lado.
    """
    events = []
    for i in np.asarray(L.get("ic", []), int):
        events.append((int(i), "L"))
    for i in np.asarray(R.get("ic", []), int):
        events.append((int(i), "R"))
    events.sort(key=lambda z: z[0])

    # Fusiona duplicados muy próximos del mismo evento y exige alternancia I/D.
    cleaned = []
    min_sep = max(1, int(round(0.18 * float(fps))))
    for idx, side in events:
        if cleaned and idx - cleaned[-1][0] < min_sep:
            # En caso de conflicto conservar el evento que mantiene alternancia.
            if side != cleaned[-1][1] and len(cleaned) >= 2 and side != cleaned[-2][1]:
                cleaned[-1] = (idx, side)
            continue
        if cleaned and side == cleaned[-1][1]:
            continue
        cleaned.append((idx, side))

    if len(cleaned) < 4:
        return {
            "cadence": np.nan, "mean_step": np.nan, "cv": np.nan, "asym": np.nan,
            "events": cleaned, "n_intervals": 0, "quality": "No fiable",
            "reason": "menos de 4 contactos alternantes válidos"
        }

    frames = np.asarray([e[0] for e in cleaned], float)
    sides = [e[1] for e in cleaned]
    intervals = np.diff(frames) / float(fps)
    med = float(np.nanmedian(intervals))

    # Solo elimina intervalos manifiestamente incompatibles con un paso;
    # no recorta variabilidad clínica moderada.
    good = np.isfinite(intervals) & (intervals >= max(0.18, 0.45 * med)) & (intervals <= min(2.0, 1.80 * med))
    intervals2 = intervals[good]
    transitions = [(sides[i], sides[i+1]) for i in range(len(intervals)) if good[i]]

    if len(intervals2) < 3 or np.nanmean(intervals2) <= 0:
        return {
            "cadence": np.nan, "mean_step": np.nan, "cv": np.nan, "asym": np.nan,
            "events": cleaned, "n_intervals": int(len(intervals2)), "quality": "No fiable",
            "reason": "insuficientes intervalos de paso plausibles"
        }

    mean_step = float(np.nanmean(intervals2))
    cadence = float(60.0 / mean_step)
    cv = float(np.nanstd(intervals2, ddof=1) / mean_step * 100.0) if len(intervals2) >= 3 else np.nan

    lr = [v for v, tr in zip(intervals2, transitions) if tr == ("L", "R")]
    rl = [v for v, tr in zip(intervals2, transitions) if tr == ("R", "L")]
    if len(lr) >= 2 and len(rl) >= 2:
        ml, mr = float(np.mean(lr)), float(np.mean(rl))
        asym = abs(ml - mr) / ((ml + mr) / 2.0) * 100.0 if (ml + mr) > 0 else np.nan
    else:
        asym = np.nan

    quality = "Alta" if len(intervals2) >= 6 else "Moderada"
    return {
        "cadence": cadence, "mean_step": mean_step, "cv": cv, "asym": asym,
        "events": cleaned, "n_intervals": int(len(intervals2)), "quality": quality,
        "reason": f"{len(intervals2)} intervalos derivados de contactos I/D validados"
    }





def _kinematic_alternation_metrics(seg, fps):
    """
    v0.10.6 · Ritmo independiente de las máscaras de apoyo.

    Usa exclusivamente trayectorias articulares distales ya trackeadas:
      señal = posición vertical relativa pie derecho - pie izquierdo
    normalizada por la altura aparente del tronco y detrendida lentamente.

    Los máximos/mínimos alternantes representan posiciones relativas sucesivas
    de avance de una y otra extremidad. No equivalen a footswitch/GRF, pero son
    apropiados como detector cinemático de periodicidad cuando el tracking es
    bueno y las máscaras de contacto fallan.

    Devuelve:
      - cadence: pasos/min
      - asym: asimetría temporal de alternancia (%)
      - cv: CV robusto del intervalo de alternancia (%)
      - events: posiciones dentro de seg
      - event_frames: números de frame reales
    """
    if seg is None or len(seg) < max(45, int(round(2.5*fps))):
        return {
            "cadence": np.nan, "asym": np.nan, "cv": np.nan,
            "events": np.asarray([], int), "event_frames": np.asarray([], int),
            "intervals": np.asarray([], float), "n_intervals": 0,
            "quality": "No fiable", "periodicity": np.nan,
            "reason": "segmento insuficiente"
        }

    try:
        ly = (
            pd.to_numeric(seg["LAnkle_y"], errors="coerce").to_numpy(float)
            + pd.to_numeric(seg["LHeel_y"], errors="coerce").to_numpy(float)
            + pd.to_numeric(seg["LBigToe_y"], errors="coerce").to_numpy(float)
        ) / 3.0
        ry = (
            pd.to_numeric(seg["RAnkle_y"], errors="coerce").to_numpy(float)
            + pd.to_numeric(seg["RHeel_y"], errors="coerce").to_numpy(float)
            + pd.to_numeric(seg["RBigToe_y"], errors="coerce").to_numpy(float)
        ) / 3.0

        hip_y = (
            pd.to_numeric(seg["LHip_y"], errors="coerce").to_numpy(float)
            + pd.to_numeric(seg["RHip_y"], errors="coerce").to_numpy(float)
        ) / 2.0
        sh_y = (
            pd.to_numeric(seg["LShoulder_y"], errors="coerce").to_numpy(float)
            + pd.to_numeric(seg["RShoulder_y"], errors="coerce").to_numpy(float)
        ) / 2.0
        torso = np.abs(hip_y-sh_y)
        torso_med = float(np.nanmedian(torso[np.isfinite(torso) & (torso > 5)])) if np.any(np.isfinite(torso) & (torso > 5)) else np.nan
        denom = np.where(np.isfinite(torso) & (torso > 5), torso, torso_med if np.isfinite(torso_med) else 1.0)

        sig = (ry-ly) / denom
        sig = (
            pd.Series(sig)
            .interpolate(limit_direction="both")
            .rolling(max(3, int(round(0.22*fps))) | 1, center=True, min_periods=1)
            .mean()
            .to_numpy(float)
        )

        # Eliminar deriva por perspectiva/aproximación a cámara sin borrar pasos lentos.
        trend_w = max(15, int(round(2.2*fps)))
        if trend_w % 2 == 0:
            trend_w += 1
        trend = pd.Series(sig).rolling(trend_w, center=True, min_periods=1).mean().to_numpy(float)
        z = sig - trend

        amp = float(np.nanstd(z))
        if not np.isfinite(amp) or amp < 0.008:
            return {
                "cadence": np.nan, "asym": np.nan, "cv": np.nan,
                "events": np.asarray([], int), "event_frames": np.asarray([], int),
                "intervals": np.asarray([], float), "n_intervals": 0,
                "quality": "No fiable", "periodicity": np.nan,
                "reason": "amplitud de alternancia insuficiente"
            }

        # Un pico del mismo signo no puede repetirse demasiado pronto.
        distance = max(3, int(round(0.40*fps)))
        prominence = max(0.012, 0.34*amp)

        peaks, pprop = find_peaks(z, distance=distance, prominence=prominence)
        troughs, tprop = find_peaks(-z, distance=distance, prominence=prominence)

        candidates = []
        for k, i in enumerate(peaks):
            candidates.append((int(i), 1, float(pprop["prominences"][k])))
        for k, i in enumerate(troughs):
            candidates.append((int(i), -1, float(tprop["prominences"][k])))
        candidates.sort(key=lambda e: e[0])

        # Exigir alternancia de signo. Si aparecen dos extremos iguales seguidos,
        # conservar el más prominente.
        events = []
        for ev in candidates:
            if not events:
                events.append(ev)
                continue
            if ev[1] == events[-1][1]:
                if ev[2] > events[-1][2]:
                    events[-1] = ev
            else:
                events.append(ev)

        if len(events) < 5:
            return {
                "cadence": np.nan, "asym": np.nan, "cv": np.nan,
                "events": np.asarray([e[0] for e in events], int),
                "event_frames": np.asarray([], int),
                "intervals": np.asarray([], float), "n_intervals": max(0,len(events)-1),
                "quality": "No fiable", "periodicity": np.nan,
                "reason": "menos de 5 extremos alternantes"
            }

        pos = np.asarray([e[0] for e in events], int)
        dirs = np.asarray([e[1] for e in events], int)
        dt = np.diff(pos) / float(fps)
        trans_dir = dirs[:-1]  # +1→-1 o -1→+1

        # Límites muy amplios para marcha neurológica lenta.
        plausible = np.isfinite(dt) & (dt >= 0.30) & (dt <= 2.50)
        dt0 = dt[plausible]
        td0 = trans_dir[plausible]

        if len(dt0) < 4:
            return {
                "cadence": np.nan, "asym": np.nan, "cv": np.nan,
                "events": pos, "event_frames": np.asarray([], int),
                "intervals": dt0, "n_intervals": int(len(dt0)),
                "quality": "No fiable", "periodicity": np.nan,
                "reason": "menos de 4 intervalos de alternancia plausibles"
            }

        # Rechazo robusto suave: preserva variabilidad patológica real.
        med = float(np.nanmedian(dt0))
        mad = float(np.nanmedian(np.abs(dt0-med)))
        keep = np.ones(len(dt0), dtype=bool)
        if len(dt0) >= 7 and mad > 1e-6:
            rz = 0.6745*np.abs(dt0-med)/mad
            keep &= rz <= 4.5
        keep &= (dt0 >= max(0.30, 0.45*med)) & (dt0 <= min(2.50, 2.20*med))

        dtk = dt0[keep]
        tdk = td0[keep]
        if len(dtk) < 4:
            dtk, tdk = dt0, td0

        # Mediana = cadencia robusta frente a una pausa o paso muy prolongado.
        step_med = float(np.nanmedian(dtk))
        cadence = float(60.0/step_med) if step_med > 0 else np.nan

        # CV robusto con SD clásica sobre intervalos ya depurados.
        cv = (
            float(np.nanstd(dtk, ddof=1)/np.nanmean(dtk)*100.0)
            if len(dtk) >= 3 and np.nanmean(dtk) > 0 else np.nan
        )

        a = dtk[tdk > 0]
        b = dtk[tdk < 0]
        asym = np.nan
        if len(a) >= 2 and len(b) >= 2:
            ma, mb = float(np.nanmedian(a)), float(np.nanmedian(b))
            den = (ma+mb)/2.0
            asym = abs(ma-mb)/den*100.0 if den > 0 else np.nan

        # Periodicidad independiente mediante autocorrelación: pico de zancada.
        zz = np.nan_to_num(z - np.nanmean(z))
        ac = np.correlate(zz, zz, mode="full")[len(zz)-1:]
        periodicity = np.nan
        if len(ac) and ac[0] > 1e-12:
            ac = ac/ac[0]
            lo = max(2, int(round(0.65*fps)))
            hi = min(len(ac)-2, int(round(5.0*fps)))
            loc = []
            for i in range(lo+1, hi):
                if ac[i] > ac[i-1] and ac[i] >= ac[i+1]:
                    loc.append(i)
            if loc:
                best = max(loc, key=lambda i: ac[i])
                periodicity = float(ac[best])

        nint = int(len(dtk))
        if nint >= 10 and (not np.isfinite(periodicity) or periodicity >= 0.10):
            quality = "Alta"
        elif nint >= 6:
            quality = "Moderada"
        else:
            quality = "Baja"

        frame_vals = pd.to_numeric(seg["frame"], errors="coerce").to_numpy()
        event_frames = []
        for p in pos:
            if 0 <= p < len(frame_vals) and np.isfinite(frame_vals[p]):
                event_frames.append(int(frame_vals[p]))

        return {
            "cadence": cadence,
            "asym": asym,
            "cv": cv,
            "events": pos,
            "event_frames": np.asarray(event_frames, int),
            "intervals": dtk,
            "n_intervals": nint,
            "quality": quality,
            "periodicity": periodicity,
            "reason": f"{nint} intervalos de alternancia distal válidos"
        }
    except Exception as e:
        return {
            "cadence": np.nan, "asym": np.nan, "cv": np.nan,
            "events": np.asarray([], int), "event_frames": np.asarray([], int),
            "intervals": np.asarray([], float), "n_intervals": 0,
            "quality": "No fiable", "periodicity": np.nan,
            "reason": f"error detector cinemático: {e}"
        }


def _canonical_gait_timeline(L, R, fps):
    """
    v0.9.2 · Línea temporal anatómica canónica.

    Construye una única secuencia temporal de contactos iniciales (IC)
    izquierdos/derechos procedentes de LOS MISMOS ciclos IC→TO→IC usados
    para apoyo/oscilación.

    Principios:
    - exige alternancia anatómica L-R-L-R;
    - no usa pares/impares como sustituto de lateralidad;
    - prioriza intervalos próximos a 1/2 del ciclo ipsilateral mediano;
    - conserva la secuencia temporal de mayor coherencia;
    - no elimina un ciclo únicamente porque aumente el CV.
    """
    fps = float(max(fps, 1.0))
    events = []
    for i in np.asarray(L.get("ic", []), dtype=int):
        events.append((int(i), "L"))
    for i in np.asarray(R.get("ic", []), dtype=int):
        events.append((int(i), "R"))
    events.sort(key=lambda z: (z[0], z[1]))

    raw_cycles = np.r_[
        np.asarray(L.get("cycle", []), dtype=float),
        np.asarray(R.get("cycle", []), dtype=float)
    ]
    raw_cycles = raw_cycles[np.isfinite(raw_cycles) & (raw_cycles >= 0.55) & (raw_cycles <= 2.50)]
    stride_ref = float(np.nanmedian(raw_cycles)) if len(raw_cycles) else np.nan
    step_ref = stride_ref / 2.0 if np.isfinite(stride_ref) and stride_ref > 0 else np.nan

    if len(events) < 4:
        return {
            "events": events, "intervals": np.asarray([], float),
            "cadence": np.nan, "mean_step": np.nan,
            "asym": np.nan, "lr_mean": np.nan, "rl_mean": np.nan,
            "n_intervals": 0, "quality": "No fiable",
            "reason": "menos de 4 IC anatómicos disponibles",
            "stride_ref": stride_ref,
        }

    # Elimina duplicados del mismo lado extremadamente próximos.
    dedup = []
    duplicate_gap = max(1, int(round(0.16 * fps)))
    for ev in events:
        if dedup and ev[1] == dedup[-1][1] and ev[0] - dedup[-1][0] < duplicate_gap:
            continue
        dedup.append(ev)
    events = dedup

    # Intervalo de paso plausible. Es amplio para no borrar variabilidad clínica,
    # pero evita saltos manifiestamente incompatibles con un paso.
    if np.isfinite(step_ref):
        lo_step = max(0.22, 0.58 * step_ref)
        hi_step = min(1.35, 1.55 * step_ref)
    else:
        lo_step, hi_step = 0.22, 1.35

    # Programación dinámica: mejor cadena alternante.
    # score alto = más eventos + menor desviación respecto al paso de referencia.
    n = len(events)
    best_score = np.ones(n, dtype=float)
    prev_idx = np.full(n, -1, dtype=int)
    chain_len = np.ones(n, dtype=int)

    for j in range(n):
        fj, sj = events[j]
        for i in range(j):
            fi, si = events[i]
            if si == sj:
                continue
            dt = (fj - fi) / fps
            if not (lo_step <= dt <= hi_step):
                continue
            penalty = 0.0
            if np.isfinite(step_ref) and step_ref > 0:
                penalty = min(abs(dt - step_ref) / step_ref, 1.0)
            candidate = best_score[i] + 1.0 - 0.18 * penalty
            if candidate > best_score[j]:
                best_score[j] = candidate
                prev_idx[j] = i
                chain_len[j] = chain_len[i] + 1

    end = int(np.argmax(best_score + 0.02 * chain_len))
    idxs = []
    k = end
    while k >= 0:
        idxs.append(k)
        k = int(prev_idx[k])
    idxs = idxs[::-1]
    chain = [events[i] for i in idxs]

    if len(chain) < 4:
        return {
            "events": chain, "intervals": np.asarray([], float),
            "cadence": np.nan, "mean_step": np.nan,
            "asym": np.nan, "lr_mean": np.nan, "rl_mean": np.nan,
            "n_intervals": max(0, len(chain)-1), "quality": "No fiable",
            "reason": "no se obtuvo una cadena L-R alternante suficientemente larga",
            "stride_ref": stride_ref,
        }

    frames = np.asarray([e[0] for e in chain], dtype=float)
    sides = [e[1] for e in chain]
    intervals = np.diff(frames) / fps
    transitions = [(sides[i], sides[i+1]) for i in range(len(sides)-1)]

    good = np.isfinite(intervals) & (intervals >= lo_step) & (intervals <= hi_step)
    intervals_good = intervals[good]
    trans_good = [tr for tr, ok in zip(transitions, good) if ok]

    if len(intervals_good) < 3 or float(np.nanmean(intervals_good)) <= 0:
        return {
            "events": chain, "intervals": intervals_good,
            "cadence": np.nan, "mean_step": np.nan,
            "asym": np.nan, "lr_mean": np.nan, "rl_mean": np.nan,
            "n_intervals": int(len(intervals_good)), "quality": "No fiable",
            "reason": "menos de 3 intervalos de paso anatómicos plausibles",
            "stride_ref": stride_ref,
        }

    mean_step = float(np.nanmean(intervals_good))
    cadence = float(60.0 / mean_step)

    lr = np.asarray([v for v, tr in zip(intervals_good, trans_good) if tr == ("L","R")], float)
    rl = np.asarray([v for v, tr in zip(intervals_good, trans_good) if tr == ("R","L")], float)
    lr_mean = float(np.nanmean(lr)) if len(lr) else np.nan
    rl_mean = float(np.nanmean(rl)) if len(rl) else np.nan

    if len(lr) >= 2 and len(rl) >= 2 and np.isfinite(lr_mean) and np.isfinite(rl_mean):
        denom = (lr_mean + rl_mean) / 2.0
        asym = abs(lr_mean - rl_mean) / denom * 100.0 if denom > 0 else np.nan
    else:
        asym = np.nan

    if len(intervals_good) >= 7 and len(lr) >= 3 and len(rl) >= 3:
        quality = "Alta"
    elif len(intervals_good) >= 5:
        quality = "Moderada"
    else:
        quality = "Baja"

    return {
        "events": chain,
        "intervals": intervals_good,
        "cadence": cadence,
        "mean_step": mean_step,
        "asym": asym,
        "lr_mean": lr_mean,
        "rl_mean": rl_mean,
        "n_lr": int(len(lr)),
        "n_rl": int(len(rl)),
        "n_intervals": int(len(intervals_good)),
        "quality": quality,
        "reason": f"{len(intervals_good)} pasos en cadena anatómica alternante",
        "stride_ref": stride_ref,
    }


def _raw_stride_cadence(L, R):
    """
    Cadencia de control desde ciclos ipsilaterales SIN usar el filtro del CV.
    Así un ciclo estadísticamente variable no altera silenciosamente el ritmo.
    """
    vals = np.r_[
        np.asarray(L.get("cycle", []), dtype=float),
        np.asarray(R.get("cycle", []), dtype=float)
    ]
    vals = vals[np.isfinite(vals) & (vals >= 0.55) & (vals <= 2.50)]
    if len(vals) < 2:
        return np.nan, np.nan, int(len(vals))
    mean_stride = float(np.nanmean(vals))
    cadence = float(120.0 / mean_stride) if mean_stride > 0 else np.nan
    return cadence, mean_stride, int(len(vals))


def _temporal_closure_from_masks(lmask, rmask, L, R, straight_mask, fps):
    """
    Control físico independiente v0.9.2.

    Para marcha sin fase aérea relevante:
        stance_L + stance_R = ciclo * (1 + doble_apoyo_fracción)

    Permite estimar qué cadencia sería compatible simultáneamente con
    tiempos de apoyo y solapamiento bipodal observados.
    """
    cov_l = np.asarray(L.get("coverage_mask", np.ones(len(lmask), dtype=bool)), bool)
    cov_r = np.asarray(R.get("coverage_mask", np.ones(len(rmask), dtype=bool)), bool)
    valid = cov_l & cov_r & np.asarray(straight_mask, bool)

    if not valid.any():
        return {
            "double_raw": np.nan, "flight_pct": np.nan,
            "stance_pct_l": np.nan, "stance_pct_r": np.nan,
            "double_expected": np.nan, "double_discrepancy": np.nan,
            "closure_stride": np.nan, "closure_cadence": np.nan,
        }

    both = np.asarray(lmask, bool) & np.asarray(rmask, bool)
    none = (~np.asarray(lmask, bool)) & (~np.asarray(rmask, bool))
    double_raw = float(np.mean(both[valid]) * 100.0)
    flight_pct = float(np.mean(none[valid]) * 100.0)

    st_l = float(np.nanmean(L.get("stance", []))) if len(L.get("stance", [])) else np.nan
    st_r = float(np.nanmean(R.get("stance", []))) if len(R.get("stance", [])) else np.nan
    sp_l = float(np.nanmean(L.get("stance_pct", []))) if len(L.get("stance_pct", [])) else np.nan
    sp_r = float(np.nanmean(R.get("stance_pct", []))) if len(R.get("stance_pct", [])) else np.nan

    double_expected = max(0.0, sp_l + sp_r - 100.0) if np.isfinite(sp_l) and np.isfinite(sp_r) else np.nan
    double_discrepancy = abs(double_raw - double_expected) if np.isfinite(double_raw) and np.isfinite(double_expected) else np.nan

    closure_stride = np.nan
    closure_cadence = np.nan
    if (
        np.isfinite(st_l) and np.isfinite(st_r) and
        np.isfinite(double_raw) and
        np.isfinite(flight_pct) and flight_pct <= 5.0
    ):
        denom = 1.0 + double_raw / 100.0
        if denom > 0:
            closure_stride = float((st_l + st_r) / denom)
            if closure_stride > 0:
                closure_cadence = float(120.0 / closure_stride)

    return {
        "double_raw": double_raw,
        "flight_pct": flight_pct,
        "stance_pct_l": sp_l,
        "stance_pct_r": sp_r,
        "double_expected": double_expected,
        "double_discrepancy": double_discrepancy,
        "closure_stride": closure_stride,
        "closure_cadence": closure_cadence,
    }


def _pct_disagreement(a, b):
    if not (np.isfinite(a) and np.isfinite(b)) or a <= 0 or b <= 0:
        return np.nan
    return float(abs(a-b) / ((a+b)/2.0) * 100.0)


def _cycle_timing_metrics(L, R, fps):
    """
    v0.9.2 · Estimador robusto de VARIABILIDAD por lado.

    Objetivos:
    - Estimar el CV sin mezclar directamente ambas piernas.
    - Evitar inflar el CV mezclando directamente ciclos izquierdos y derechos.
    - Rechazar ciclos atípicos de forma robusta por lado.
    - Informar tamaño muestral y confianza.
    """

    def _robust_side(x):
        x = np.asarray(x, dtype=float)
        x = x[np.isfinite(x) & (x > 0)]
        if len(x) == 0:
            return {
                "raw": x, "clean": x, "mean": np.nan, "sd": np.nan, "cv": np.nan,
                "n_raw": 0, "n_clean": 0, "outliers": 0
            }

        # Filtro fisiológicamente amplio inicial.
        x = x[(x >= 0.55) & (x <= 2.50)]
        if len(x) == 0:
            return {
                "raw": x, "clean": x, "mean": np.nan, "sd": np.nan, "cv": np.nan,
                "n_raw": 0, "n_clean": 0, "outliers": 0
            }

        raw = x.copy()
        med = float(np.nanmedian(x))
        mad = float(np.nanmedian(np.abs(x - med)))

        if len(x) >= 4 and mad > 1e-9:
            robust_z = 0.6745 * np.abs(x - med) / mad
            keep = robust_z <= 3.5
            x = x[keep]

        # Segundo filtro relativo conservador para evitar falsos IC→IC.
        if len(x) >= 3:
            med2 = float(np.nanmedian(x))
            lo = max(0.55, 0.75 * med2)
            hi = min(2.50, 1.25 * med2)
            x = x[(x >= lo) & (x <= hi)]

        mean = float(np.nanmean(x)) if len(x) else np.nan
        sd = float(np.nanstd(x, ddof=1)) if len(x) >= 2 else np.nan
        cv = float(sd / mean * 100.0) if np.isfinite(sd) and np.isfinite(mean) and mean > 0 else np.nan

        return {
            "raw": raw,
            "clean": x,
            "mean": mean,
            "sd": sd,
            "cv": cv,
            "n_raw": int(len(raw)),
            "n_clean": int(len(x)),
            "outliers": int(max(0, len(raw) - len(x))),
        }

    left = _robust_side(L.get("cycle", []))
    right = _robust_side(R.get("cycle", []))

    # Cadencia global desde la duración media de zancada por lado.
    side_means = [v for v in (left["mean"], right["mean"]) if np.isfinite(v)]
    if len(side_means) == 0:
        return {
            "cadence": np.nan, "mean_stride": np.nan, "cv": np.nan,
            "cv_left": np.nan, "cv_right": np.nan,
            "l_mean_stride": np.nan, "r_mean_stride": np.nan,
            "n_left": left["n_clean"], "n_right": right["n_clean"],
            "n_cycles": left["n_clean"] + right["n_clean"],
            "outliers_left": left["outliers"], "outliers_right": right["outliers"],
            "quality": "No fiable",
            "reason": "sin ciclos IC→IC válidos"
        }

    # Media ponderada por número de ciclos válidos.
    num = 0.0
    den = 0
    for side in (left, right):
        if np.isfinite(side["mean"]) and side["n_clean"] > 0:
            num += side["mean"] * side["n_clean"]
            den += side["n_clean"]
    mean_stride = float(num / den) if den > 0 else float(np.nanmean(side_means))
    cadence = float(120.0 / mean_stride) if mean_stride > 0 else np.nan

    # CV global robusto = promedio ponderado de CV izquierdo y derecho.
    cv_num = 0.0
    cv_den = 0
    for side in (left, right):
        if np.isfinite(side["cv"]) and side["n_clean"] >= 2:
            cv_num += side["cv"] * side["n_clean"]
            cv_den += side["n_clean"]
    cv_global = float(cv_num / cv_den) if cv_den > 0 else np.nan

    nL, nR = left["n_clean"], right["n_clean"]
    n_total = nL + nR

    # Confianza explícita según tamaño muestral bilateral.
    if nL >= 4 and nR >= 4:
        quality = "Alta"
    elif nL >= 3 and nR >= 3:
        quality = "Moderada"
    elif nL >= 2 and nR >= 2:
        quality = "Baja"
    else:
        quality = "Muy baja"

    return {
        "cadence": cadence,
        "mean_stride": mean_stride,
        "cv": cv_global,
        "cv_left": left["cv"],
        "cv_right": right["cv"],
        "l_mean_stride": left["mean"],
        "r_mean_stride": right["mean"],
        "n_left": nL,
        "n_right": nR,
        "n_cycles": n_total,
        "outliers_left": left["outliers"],
        "outliers_right": right["outliers"],
        "quality": quality,
        "reason": (
            f"{nL} ciclos izquierdos + {nR} derechos tras filtrado robusto; "
            f"outliers excluidos L={left['outliers']}, R={right['outliers']}"
        )
    }


def _run_durations(mask, fps, min_s=0.15, max_s=10.0):
    vals = []
    for a, b in _bool_runs(mask):
        d = (b - a) / float(fps)
        if min_s <= d <= max_s:
            vals.append(d)
    return np.asarray(vals, float)


def _medial_knee_deviation_angle(row, side):
    hx,hy=row[f"{side}Hip_x"],row[f"{side}Hip_y"]
    kx,ky=row[f"{side}Knee_x"],row[f"{side}Knee_y"]
    ax,ay=row[f"{side}Ankle_x"],row[f"{side}Ankle_y"]
    midx=(row.LHip_x+row.RHip_x)/2.0
    if not all(np.isfinite([hx,hy,kx,ky,ax,ay,midx])) or abs(ay-hy)<1e-6:
        return np.nan
    t=(ky-hy)/(ay-hy)
    line_x=hx+t*(ax-hx)
    toward_mid=(kx-line_x)*(midx-line_x)>0
    if not toward_mid:
        return 0.0
    ka=point_angle(hx,hy,kx,ky,ax,ay)
    return abs(180.0-ka) if np.isfinite(ka) else np.nan

def add_frontal_advanced(seg, fps, scale_cm_per_px=0.0):
    seg=seg.copy()
    hipx=(seg.LHip_x.to_numpy(float)+seg.RHip_x.to_numpy(float))/2.0
    shx=(seg.LShoulder_x.to_numpy(float)+seg.RShoulder_x.to_numpy(float))/2.0
    # Proxy del centro de masa: combinación tronco-pelvis, no CoM segmentario 3D.
    seg["com_proxy_x"] = 0.65*hipx + 0.35*shx
    lmask, _, _ = _support_mask_2d(seg, "L", fps); rmask, _, _ = _support_mask_2d(seg, "R", fps)
    seg["L_support_2d"]=lmask; seg["R_support_2d"]=rmask
    lx,_=_foot_centroid(seg,"L"); rx,_=_foot_centroid(seg,"R")
    seg["bos_px"]=np.abs(rx-lx)
    # v25 · COM (Centro de Masa estimado 2D) ampliado. Sigue siendo un proxy tronco-pelvis, no un COM 3D segmentario.
    com_center = seg["com_proxy_x"] - np.nanmedian(seg["com_proxy_x"])
    pelvis_width_px = np.abs(seg.RHip_x.to_numpy(float) - seg.LHip_x.to_numpy(float))
    pelvis_med = float(np.nanmedian(pelvis_width_px[np.isfinite(pelvis_width_px) & (pelvis_width_px > 1)])) if np.any(np.isfinite(pelvis_width_px) & (pelvis_width_px > 1)) else np.nan
    seg["com_proxy_centered_px"] = com_center
    seg["com_proxy_norm_pct_pelvis"] = (com_center / pelvis_med * 100.0) if np.isfinite(pelvis_med) and pelvis_med > 0 else np.nan
    # Posición del COM respecto al centro de la BOS: 0=centro; ±1≈centro de uno de los pies.
    bos_mid = (lx + rx) / 2.0
    half_bos = np.abs(rx-lx) / 2.0
    seg["com_bos_ratio"] = np.divide(seg.com_proxy_x.to_numpy(float)-bos_mid, half_bos, out=np.full(len(seg), np.nan), where=half_bos>2.0)
    # Velocidad mediolateral en px/s; útil longitudinalmente con geometría de cámara idéntica.
    seg["com_velocity_px_s"] = np.gradient(pd.Series(com_center).interpolate(limit_direction="both").to_numpy(float)) * float(fps)
    seg["L_dynamic_valgus"]=[_medial_knee_deviation_angle(r,"L") for _,r in seg.iterrows()]
    seg["R_dynamic_valgus"]=[_medial_knee_deviation_angle(r,"R") for _,r in seg.iterrows()]
    if scale_cm_per_px and scale_cm_per_px>0:
        seg["com_proxy_cm"]=(seg.com_proxy_x-np.nanmedian(seg.com_proxy_x))*float(scale_cm_per_px)
        seg["bos_cm"]=seg.bos_px*float(scale_cm_per_px)
    else:
        seg["com_proxy_cm"]=np.nan
        seg["bos_cm"]=np.nan
    return seg

def visibility_pct(seg, names, threshold=0.5):
    cols = [f"{n}_score" for n in names if f"{n}_score" in seg.columns]
    if not cols:
        return np.nan
    return float((seg[cols].min(axis=1) >= threshold).mean() * 100)


def _sync_signal(df):
    """Señal corporal vertical normalizada para estimar desfase temporal entre cámaras."""
    if df is None or df.empty:
        return np.array([]), np.array([])
    hip_y = (df["LHip_y"].to_numpy(float) + df["RHip_y"].to_numpy(float)) / 2.0
    sh_y = (df["LShoulder_y"].to_numpy(float) + df["RShoulder_y"].to_numpy(float)) / 2.0
    torso = np.abs(hip_y - sh_y)
    torso[torso < 1.0] = np.nan
    sig = hip_y / torso
    sig = pd.Series(sig).interpolate(limit_direction="both").rolling(7, center=True, min_periods=1).mean().to_numpy()
    sig = np.gradient(sig)
    return df["frame"].to_numpy(float), sig


def estimate_sync_offset(df1, fps1, df2, fps2, max_offset_s=2.0, resample_hz=50):
    """
    Estima desfase cam02 vs cam01 por correlación de movimiento vertical corporal.
    offset_s > 0: el mismo evento aparece más tarde en cam02; para alinear, avanzar/recortar cam02 ese tiempo.
    Es una heurística experimental, no sustituto de sincronización hardware.
    """
    f1, s1 = _sync_signal(df1); f2, s2 = _sync_signal(df2)
    if len(s1) < 20 or len(s2) < 20 or fps1 <= 0 or fps2 <= 0:
        return 0.0, np.nan, "No calculable"
    t1 = f1 / float(fps1); t2 = f2 / float(fps2)
    dur = min(float(np.nanmax(t1)), float(np.nanmax(t2)))
    if dur < 2.0:
        return 0.0, np.nan, "No calculable"
    grid = np.arange(0.0, dur, 1.0 / resample_hz)
    a = np.interp(grid, t1, s1); b = np.interp(grid, t2, s2)
    a = (a - np.nanmean(a)) / (np.nanstd(a) + 1e-8)
    b = (b - np.nanmean(b)) / (np.nanstd(b) + 1e-8)
    max_k = int(round(max_offset_s * resample_hz))
    best_k, best_corr = 0, -np.inf
    for k in range(-max_k, max_k + 1):
        if k > 0:
            x, y = a[:-k], b[k:]
        elif k < 0:
            x, y = a[-k:], b[:k]
        else:
            x, y = a, b
        if len(x) < resample_hz:
            continue
        c = np.corrcoef(x, y)[0,1]
        if np.isfinite(c) and c > best_corr:
            best_corr, best_k = float(c), int(k)
    offset_s = best_k / float(resample_hz)
    quality = "Alta" if best_corr >= 0.65 else ("Moderada" if best_corr >= 0.40 else "Baja")
    return float(offset_s), float(best_corr), quality



def _visibility_signal(df, threshold=0.35):
    """Versión 31 · visibilidad corporal por frame para localizar el tramo útil de cada cámara."""
    if df is None or df.empty:
        return np.asarray([], dtype=float)
    score_cols=[c for c in df.columns if c.endswith("_score")]
    if not score_cols:
        return np.ones(len(df), dtype=float)
    a=df[score_cols].apply(pd.to_numeric, errors="coerce").to_numpy(float)
    return np.nanmean(a >= float(threshold), axis=1).astype(float)


def _active_visibility_window(df, fps, min_visible_fraction=0.30, pad_s=0.20):
    """Devuelve inicio/fin temporal del intervalo continuo principal donde el sujeto es utilizable."""
    if df is None or df.empty or fps <= 0:
        return None
    vis=_visibility_signal(df)
    if len(vis)==0:
        return None
    # suavizado para tolerar oclusiones breves
    win=max(3, int(round(float(fps)*0.25)))
    sm=pd.Series(vis).rolling(win, center=True, min_periods=1).mean().to_numpy()
    mask=sm >= float(min_visible_fraction)
    starts,ends=_edges(mask)
    if not len(starts):
        return None
    spans=[(int(a),int(b)) for a,b in zip(starts,ends) if b-a >= max(5,int(0.4*fps))]
    if not spans:
        return None
    a,b=max(spans,key=lambda z:z[1]-z[0])
    pad=int(round(pad_s*fps))
    a=max(0,a-pad); b=min(len(df),b+pad)
    frames=pd.to_numeric(df.get("frame",pd.Series(np.arange(len(df)))),errors="coerce").to_numpy(float)
    fa=frames[a] if a<len(frames) and np.isfinite(frames[a]) else a
    ib=min(max(a,b-1),len(frames)-1)
    fb=frames[ib] if np.isfinite(frames[ib]) else ib
    return {"start_frame":int(round(fa)),"end_frame":int(round(fb)),"start_s":float(fa/fps),"end_s":float(fb/fps),"duration_s":float(max(0,(fb-fa)/fps))}


def estimate_biplanar_sync(df1, fps1, df2, fps2, resample_hz=40):
    """
    Versión 31 · sincronización para el montaje clínico frontal+lateral.
    Busca la señal vertical de la cámara lateral dentro de toda la señal frontal,
    por lo que tolera que cam02 solo vea al paciente durante la zona central.
    offset > 0 significa t_cam02 = t_cam01 + offset para el mismo instante físico.
    """
    w1=_active_visibility_window(df1,fps1); w2=_active_visibility_window(df2,fps2)
    f1,s1=_sync_signal(df1); f2,s2=_sync_signal(df2)
    if w1 is None or w2 is None or len(s1)<20 or len(s2)<20 or fps1<=0 or fps2<=0:
        return 0.0,np.nan,"No calculable",w1,w2
    t1=f1/float(fps1); t2=f2/float(fps2)
    # Interpolar cada señal en su eje temporal original.
    g1=np.arange(float(np.nanmin(t1)),float(np.nanmax(t1))+1e-9,1/resample_hz)
    g2=np.arange(float(np.nanmin(t2)),float(np.nanmax(t2))+1e-9,1/resample_hz)
    a=np.interp(g1,t1,s1); b=np.interp(g2,t2,s2)
    a=(a-np.nanmean(a))/(np.nanstd(a)+1e-8); b=(b-np.nanmean(b))/(np.nanstd(b)+1e-8)
    # La vista lateral suele ser la ventana corta: deslizarla sobre frontal.
    # Si ocurre al revés, se evalúan offsets igualmente mediante rejilla temporal.
    offsets=np.arange(-float(np.nanmax(t1)), float(np.nanmax(t2))+1e-9, 1/resample_hz)
    best=(None,-np.inf,0)
    for off in offsets:
        # mismo instante: t2=t1+off
        lo=max(float(np.nanmin(t1)),float(np.nanmin(t2))-off)
        hi=min(float(np.nanmax(t1)),float(np.nanmax(t2))-off)
        if hi-lo < 1.2: continue
        g=np.arange(lo,hi,1/resample_hz)
        if len(g)<30: continue
        x=np.interp(g,t1,s1); y=np.interp(g+off,t2,s2)
        x=(x-np.mean(x))/(np.std(x)+1e-8); y=(y-np.mean(y))/(np.std(y)+1e-8)
        c=float(np.corrcoef(x,y)[0,1]) if len(x)>2 else np.nan
        if np.isfinite(c) and c>best[1]: best=(float(off),c,len(g))
    if best[0] is None:
        return 0.0,np.nan,"No calculable",w1,w2
    off,corr,n=best
    quality="Alta" if corr>=0.70 and n>=int(2*resample_hz) else ("Moderada" if corr>=0.45 else "Baja")
    return off,corr,quality,w1,w2



def _lateral_sagittal_event_summaries(seg, fps):
    """
    Versión 34 · detector lateral robusto de eventos de marcha.

    Combina varias señales cinemáticas 2D por extremidad:
    - posición anteroposterior del pie respecto a la pelvis,
    - velocidad horizontal relativa del pie,
    - trayectoria vertical de talón/tobillo,
    - extensión relativa de rodilla cuando está disponible,
    - alternancia temporal izquierda-derecha.

    Genera primero eventos candidatos y después acepta únicamente secuencias
    IC (Initial Contact / Contacto Inicial) -> TO (Toe Off / Despegue del Pie)
    -> siguiente IC fisiológicamente plausibles. No equivale a plataforma de
    fuerzas ni a detección cinética validada.
    """
    empty=_summary_from_positions([],[],[],fps,source="lateral_sagittal_v34")
    if seg is None or len(seg) < max(24, int(round(0.9*fps))):
        return empty, empty, {"quality":"No fiable","reason":"segmento lateral insuficiente","candidate_events":0,"accepted_events":0}

    req=["LAnkle_x","LHeel_x","LBigToe_x","RAnkle_x","RHeel_x","RBigToe_x",
         "LHip_x","RHip_x","LHip_y","RHip_y","LShoulder_y","RShoulder_y"]
    if not all(c in seg.columns for c in req):
        return empty, empty, {"quality":"No calculable","reason":"faltan trayectorias sagitales","candidate_events":0,"accepted_events":0}

    def arr(c):
        return pd.to_numeric(seg[c],errors="coerce").to_numpy(float)

    n=len(seg)
    pelvis_x=(arr("LHip_x")+arr("RHip_x"))/2.0
    torso=np.abs((arr("LHip_y")+arr("RHip_y"))/2.0-(arr("LShoulder_y")+arr("RShoulder_y"))/2.0)
    torso=pd.Series(torso).replace([np.inf,-np.inf],np.nan).interpolate(limit_direction="both").to_numpy(float)
    torso_med=float(np.nanmedian(torso)) if np.isfinite(torso).any() else np.nan
    if not np.isfinite(torso_med) or torso_med<5:
        torso_med=1.0

    # Dirección global de progresión para unificar polaridad izquierda/derecha.
    px=pd.Series(pelvis_x).replace([np.inf,-np.inf],np.nan).interpolate(limit_direction="both").to_numpy(float)
    prog=float(np.nanmedian(np.gradient(px))) if np.isfinite(px).any() else 0.0
    direction=1.0 if prog>=0 else -1.0

    ws=max(3, int(round(0.08*fps)))
    if ws % 2 == 0: ws += 1

    def smooth(x):
        return pd.Series(x).replace([np.inf,-np.inf],np.nan).interpolate(limit_direction="both").rolling(ws,center=True,min_periods=1).mean().to_numpy(float)

    # Permite usar ángulo de rodilla si ya fue calculado por add_angle_columns.
    knee_cols={"L":"L_knee_flex","R":"R_knee_flex"}

    def side_signals(side):
        ankle_x=arr(f"{side}Ankle_x"); heel_x=arr(f"{side}Heel_x"); toe_x=arr(f"{side}BigToe_x")
        foot_x=(ankle_x+heel_x+toe_x)/3.0
        rel=smooth(direction*(foot_x-pelvis_x)/torso_med)
        vel=smooth(np.gradient(rel)*float(fps))
        heel_y=smooth(arr(f"{side}Heel_y") if f"{side}Heel_y" in seg.columns else arr(f"{side}Ankle_y"))
        ankle_y=smooth(arr(f"{side}Ankle_y"))
        vert=smooth((heel_y+ankle_y)/2.0/torso_med)
        knee=None
        kc=knee_cols[side]
        if kc in seg.columns:
            knee=smooth(pd.to_numeric(seg[kc],errors="coerce").to_numpy(float))
        return rel,vel,vert,knee

    sig={s:side_signals(s) for s in ("L","R")}

    def candidate_ics(side):
        rel,vel,vert,knee=sig[side]
        if np.isfinite(rel).sum()<max(18,int(0.55*n)):
            return []
        amp=float(np.nanpercentile(rel,95)-np.nanpercentile(rel,5)) if np.isfinite(rel).any() else 0.0
        prom=max(0.10*amp,0.012)
        min_dist=max(2,int(round(0.32*fps)))
        peaks,_=find_peaks(rel,distance=min_dist,prominence=prom)
        # También aceptar máximos locales de extensión anteroposterior con poca velocidad.
        cand=[]
        vscale=float(np.nanpercentile(np.abs(vel),75)) if np.isfinite(vel).any() else 1.0
        vscale=max(vscale,1e-6)
        q95=float(np.nanpercentile(rel,75)); q50=float(np.nanmedian(rel))
        for i in peaks:
            if i<=1 or i>=n-2: continue
            pos_score=(rel[i]-q50)/(abs(q95-q50)+1e-6)
            speed_score=max(0.0,1.0-min(1.5,abs(vel[i])/vscale))/1.0
            vert_score=0.0
            # Cerca del contacto, talón/tobillo suele estar relativamente bajo en imagen;
            # solo se usa como evidencia débil porque la orientación vertical depende de cámara.
            if np.isfinite(vert[i]):
                vr=(np.nanpercentile(vert,75)-np.nanpercentile(vert,25))+1e-6
                vert_score=max(0.0,min(1.0,(np.nanpercentile(vert,75)-vert[i])/vr))
            knee_score=0.0
            if knee is not None and np.isfinite(knee[i]):
                kr=(np.nanpercentile(knee,90)-np.nanpercentile(knee,10))+1e-6
                # menor flexión = mayor extensión alrededor de IC en marcha habitual
                knee_score=max(0.0,min(1.0,(np.nanpercentile(knee,75)-knee[i])/kr+0.5))
            score=0.50*max(0.0,min(1.5,pos_score))+0.25*speed_score+0.15*vert_score+0.10*knee_score
            cand.append({"frame":int(i),"score":float(score),"side":side})
        return cand

    candidates=candidate_ics("L")+candidate_ics("R")
    candidates=sorted(candidates,key=lambda z:z["frame"])

    # Fusión temporal: conservar el mejor candidato si dos del mismo lado son casi duplicados.
    merged=[]
    duplicate_gap=max(2,int(round(0.22*fps)))
    for c in candidates:
        if merged and c["side"]==merged[-1]["side"] and c["frame"]-merged[-1]["frame"]<duplicate_gap:
            if c["score"]>merged[-1]["score"]: merged[-1]=c
        else:
            merged.append(c)

    # Coherencia de alternancia: si aparecen dos eventos consecutivos del mismo lado,
    # conservar el de mayor puntuación salvo que estén suficientemente separados como
    # para poder representar dos zancadas con ausencia de detección contralateral.
    accepted=[]
    min_step=max(2,int(round(0.28*fps))); max_step=max(min_step+1,int(round(1.25*fps)))
    for c in merged:
        if not accepted:
            accepted.append(c); continue
        dt=c["frame"]-accepted[-1]["frame"]
        if dt<min_step:
            if c["score"]>accepted[-1]["score"]: accepted[-1]=c
            continue
        if c["side"]==accepted[-1]["side"] and dt<max_step:
            if c["score"]>accepted[-1]["score"]: accepted[-1]=c
            continue
        accepted.append(c)

    by_side={"L":[],"R":[]}
    for c in accepted: by_side[c["side"]].append(int(c["frame"]))

    def build(side):
        rel,vel,vert,knee=sig[side]
        ics=np.asarray(sorted(by_side[side]),dtype=int)
        out_ic=[]; out_to=[]; out_next=[]
        for a,b in zip(ics[:-1],ics[1:]):
            cyc=(b-a)/float(fps)
            if not (0.55<=cyc<=3.20):
                continue
            # Buscar TO entre 35-85% del ciclo combinando extremo posterior del pie
            # y máximo de velocidad relativa de oscilación.
            lo=max(a+1,int(round(a+0.35*(b-a))))
            hi=min(b-1,int(round(a+0.85*(b-a))))
            if hi<=lo: continue
            rr=rel[lo:hi+1]; vv=vel[lo:hi+1]
            if not np.isfinite(rr).any(): continue
            rnorm=(rr-np.nanmin(rr))/((np.nanmax(rr)-np.nanmin(rr))+1e-6)
            vabs=np.abs(vv)
            vnorm=(vabs-np.nanmin(vabs))/((np.nanmax(vabs)-np.nanmin(vabs))+1e-6) if np.isfinite(vabs).any() else np.zeros_like(rnorm)
            # TO esperado cuando el pie inicia avance: posición relativa posterior + incremento de velocidad.
            score_to=(1.0-rnorm)*0.65+vnorm*0.35
            j=int(np.nanargmax(score_to))
            to=lo+j
            pct=(to-a)/(b-a)*100.0
            if 35.0<=pct<=85.0:
                out_ic.append(int(a)); out_to.append(int(to)); out_next.append(int(b))
        sm=_summary_from_positions(out_ic,out_to,out_next,fps,source="lateral_multisignal_v34")
        ncy=len(sm.get("cycle",[]))
        if ncy>=3: sm["score"]=92.0; sm["quality"]="Alta"
        elif ncy==2: sm["score"]=76.0; sm["quality"]="Moderada"
        elif ncy==1: sm["score"]=56.0; sm["quality"]="Baja"
        else: sm["score"]=0.0; sm["quality"]="No fiable"
        sm["reason"]=f"detector lateral multiseñal v34; {ncy} ciclos IC→IC"
        sm["candidate_ic_count"]=int(sum(1 for c in merged if c["side"]==side))
        sm["accepted_ic_count"]=int(len(ics))
        return sm

    L=build("L"); R=build("R")
    total=len(L.get("cycle",[]))+len(R.get("cycle",[]))
    ncan=len(merged); nacc=len(accepted)
    if total>=6: q="Alta"
    elif total>=3: q="Moderada"
    elif total>=1: q="Baja"
    else: q="No fiable"
    # V39 · auditoría completa de candidatos y decisiones del detector lateral.
    merged_keys={(int(c["frame"]),str(c["side"])) for c in merged}
    accepted_keys={(int(c["frame"]),str(c["side"])) for c in accepted}
    candidate_audit=[]
    for c in candidates:
        key=(int(c["frame"]),str(c["side"]))
        if key in accepted_keys:
            status="aceptado"
            reason="Conservado tras fusión temporal y filtro de alternancia/plausibilidad."
        elif key in merged_keys:
            status="rechazado_secuencia"
            reason="Superó la detección inicial pero fue descartado por coherencia temporal/alternancia."
        else:
            status="rechazado_duplicado"
            reason="Descartado durante la fusión temporal por proximidad a un candidato equivalente."
        candidate_audit.append({
            "frame_local":int(c["frame"]),"side":str(c["side"]),
            "score":float(c.get("score",np.nan)),"status":status,"reason":reason
        })
    accepted_audit=[{
        "frame_local":int(c["frame"]),"side":str(c["side"]),
        "score":float(c.get("score",np.nan)),"status":"aceptado",
        "reason":"IC lateral aceptado por el detector multiseñal."
    } for c in accepted]

    return L,R,{
        "quality":q,
        "reason":f"detector lateral multiseñal v34: {total} ciclos completos",
        "candidate_events":int(ncan),
        "accepted_events":int(nacc),
        "left_candidates":int(L.get("candidate_ic_count",0)),
        "right_candidates":int(R.get("candidate_ic_count",0)),
        "left_accepted":int(L.get("accepted_ic_count",0)),
        "right_accepted":int(R.get("accepted_ic_count",0)),
        "candidate_audit":candidate_audit,
        "accepted_audit":accepted_audit,
    }

def _make_biplanar_cycle_snapshot(df, fps, view):
    """
    Versión 33 · detecta ciclos sobre toda la ventana visible de cada cámara,
    no solo sobre la ventana de solapamiento. Después el emparejamiento decide
    qué ciclos coinciden temporalmente entre cámaras.
    """
    if df is None or len(df)==0 or not np.isfinite(fps) or fps<=0:
        return None
    w=_active_visibility_window(df,float(fps),pad_s=0.10)
    if w:
        fr=pd.to_numeric(df.get("frame",pd.Series(np.arange(len(df)))),errors="coerce")
        use=df[(fr>=w["start_frame"]) & (fr<=w["end_frame"])].copy()
    else:
        use=df.copy()
    if len(use)<max(20,int(round(1.0*fps))):
        use=df.copy()
    use=add_angle_columns(use)
    is_lat="Lateral" in (view or "")
    if is_lat:
        L,R,info=_lateral_sagittal_event_summaries(use.reset_index(drop=True),float(fps))
        # Convertir posiciones locales a números de frame reales.
        framevals=pd.to_numeric(use["frame"],errors="coerce").to_numpy(float)
        def remap(sm):
            z=dict(sm)
            for key in ("ic","to","next_ic","starts","ends"):
                vals=np.asarray(z.get(key,[]),dtype=int)
                mapped=[]
                for i in vals:
                    if 0<=i<len(framevals) and np.isfinite(framevals[i]):
                        mapped.append(int(round(framevals[i])))
                z[key]=np.asarray(mapped,dtype=int)
            return z
        L,R=remap(L),remap(R)
        # V39 · convertir también la auditoría de candidatos/aceptados a frames reales.
        info=dict(info or {})
        for audit_key in ("candidate_audit","accepted_audit"):
            remapped=[]
            for ev in info.get(audit_key,[]) or []:
                z=dict(ev)
                i=int(z.get("frame_local",-1))
                if 0<=i<len(framevals) and np.isfinite(framevals[i]):
                    z["frame"]=int(round(framevals[i]))
                    z["time_camera_s"]=float(framevals[i]/float(fps))
                remapped.append(z)
            info[audit_key]=remapped
    else:
        snap=_make_cycle_snapshot(use,float(fps))
        if snap is None:
            return None
        L,R=snap["left"],snap["right"]
        info={"quality":"Derivada","reason":"detector frontal existente sobre ventana visible completa"}
    return {"seg":use.copy(),"left":L,"right":R,"fps":float(fps),"event_info":info,"visibility_window":w}


def _cycle_overlaps_physical_window(cycle, fps, t_start, t_end, offset_s=0.0):
    """Comprueba que una parte sustancial del ciclo cae en la ventana física común."""
    if not cycle or fps<=0:
        return False
    a=float(cycle["ic_frame"])/fps-offset_s
    b=float(cycle["next_ic_frame"])/fps-offset_s
    if b<=a:
        return False
    overlap=max(0.0,min(b,t_end)-max(a,t_start))
    return overlap/(b-a)>=0.45

def _cycles_from_snapshot(snapshot):
    if not snapshot: return []
    fps=float(snapshot.get("fps",0) or 0)
    out=[]
    for side,key in [("L","left"),("R","right")]:
        for c in _build_side_cycles(snapshot.get(key) or {},fps,side):
            x=dict(c); x["ic_s"]=x["ic_frame"]/fps; x["next_ic_s"]=x["next_ic_frame"]/fps
            out.append(x)
    return sorted(out,key=lambda z:z["ic_s"])


def _biplanar_pair_hypothesis(A, B, offset_s, swap_lr=False, tolerance_s=0.24):
    """Evalúa una hipótesis anatómica L/R sin forzar correspondencias."""
    candidates=[]
    for i,a in enumerate(A):
        for j,b in enumerate(B):
            bside=b["side"]
            if swap_lr:
                bside="R" if bside=="L" else "L"
            if a["side"]!=bside:
                continue
            dt=(float(b["ic_s"])-float(offset_s))-float(a["ic_s"])
            da=float(a["next_ic_s"]-a["ic_s"]); db=float(b["next_ic_s"]-b["ic_s"])
            dd=abs(da-db)
            if abs(dt)<=float(tolerance_s) and dd<=0.35:
                candidates.append((abs(dt)+0.30*dd,i,j,dt,dd))
    candidates.sort(key=lambda z:z[0])
    pairs=[]; ua=set(); ub=set()
    for cost,i,j,dt,dd in candidates:
        if i in ua or j in ub: continue
        ua.add(i); ub.add(j)
        pairs.append({"side":A[i]["side"],"front":A[i],"lateral":B[j],
                      "delta_s":float(dt),"duration_delta_s":float(dd),
                      "cost":float(cost),"lr_swapped":bool(swap_lr)})
    pairs.sort(key=lambda p:p["front"]["ic_s"])
    mae=float(np.mean([abs(p["delta_s"]) for p in pairs])) if pairs else np.nan
    dur=float(np.mean([abs(p["duration_delta_s"]) for p in pairs])) if pairs else np.nan
    score=(len(pairs)*10.0)-(mae*12.0 if np.isfinite(mae) else 20.0)-(dur*3.0 if np.isfinite(dur) else 5.0)
    return pairs,mae,dur,float(score)

def _phase_alignment_fine_offset(front_snapshot,lateral_snapshot,coarse_offset_s,max_adjust_s=0.55):
    """
    Segunda fuente temporal independiente de IC: correlación de una señal distal continua.
    Devuelve corrección fina orientativa; no crea ciclos por sí misma.
    """
    try:
        fdf=(front_snapshot or {}).get("seg"); ldf=(lateral_snapshot or {}).get("seg")
        ffps=float((front_snapshot or {}).get("fps",0) or 0); lfps=float((lateral_snapshot or {}).get("fps",0) or 0)
        if fdf is None or ldf is None or ffps<=0 or lfps<=0: return np.nan,np.nan
        def sig(df,fps):
            # Distancia tobillos/pelvis: señal periódica robusta a traslación global.
            vals=[]
            for side in ("L","R"):
                ax=f"{side}Ankle_x"; ay=f"{side}Ankle_y"; hx=f"{side}Hip_x"; hy=f"{side}Hip_y"
                if all(c in df.columns for c in (ax,ay,hx,hy)):
                    vals.append(np.sqrt((pd.to_numeric(df[ax],errors="coerce")-pd.to_numeric(df[hx],errors="coerce"))**2+
                                        (pd.to_numeric(df[ay],errors="coerce")-pd.to_numeric(df[hy],errors="coerce"))**2).to_numpy(float))
            if not vals: return None,None
            s=np.nanmean(np.vstack(vals),axis=0)
            s=pd.Series(s).interpolate(limit_direction="both").rolling(max(3,int(round(fps*0.10))),center=True,min_periods=1).mean().to_numpy()
            t=pd.to_numeric(df["frame"],errors="coerce").to_numpy(float)/fps
            return t,(s-np.nanmean(s))/(np.nanstd(s)+1e-8)
        tf,sf=sig(fdf,ffps); tl,sl=sig(ldf,lfps)
        if tf is None or tl is None: return np.nan,np.nan
        hz=50.0; grid=np.arange(max(tf[0],tl[0]-coarse_offset_s),min(tf[-1],tl[-1]-coarse_offset_s),1/hz)
        if len(grid)<40: return np.nan,np.nan
        a=np.interp(grid,tf,sf)
        best=(np.inf,np.nan,np.nan)
        for adj in np.arange(-max_adjust_s,max_adjust_s+0.0001,1/hz):
            b=np.interp(grid+coarse_offset_s+adj,tl,sl,left=np.nan,right=np.nan)
            ok=np.isfinite(a)&np.isfinite(b)
            if ok.sum()<30: continue
            c=float(np.corrcoef(a[ok],b[ok])[0,1])
            if np.isfinite(c) and -c<best[0]: best=(-c,float(adj),c)
        return best[1],best[2]
    except Exception:
        return np.nan,np.nan


def _v38_master_clock_signal(snapshot):
    """
    V38 · señal corporal común para sincronización maestra independiente de IC.
    Combina pelvis, tronco y segmentos distales normalizados. No usa IC/TO.
    """
    try:
        df=(snapshot or {}).get("seg")
        fps=float((snapshot or {}).get("fps",0) or 0)
        if df is None or len(df)<20 or fps<=0:
            return None,None
        frame=pd.to_numeric(df["frame"],errors="coerce").to_numpy(float)
        t=frame/fps
        comps=[]
        # Pelvis: centro de caderas
        if all(c in df.columns for c in ("LHip_y","RHip_y")):
            comps.append((pd.to_numeric(df["LHip_y"],errors="coerce").to_numpy(float)+
                          pd.to_numeric(df["RHip_y"],errors="coerce").to_numpy(float))/2.0)
        # Tronco/hombros
        if all(c in df.columns for c in ("LShoulder_y","RShoulder_y")):
            comps.append((pd.to_numeric(df["LShoulder_y"],errors="coerce").to_numpy(float)+
                          pd.to_numeric(df["RShoulder_y"],errors="coerce").to_numpy(float))/2.0)
        # Rodillas y tobillos: media bilateral, útil para periodicidad global
        for joint in ("Knee","Ankle"):
            cols=(f"L{joint}_y",f"R{joint}_y")
            if all(c in df.columns for c in cols):
                comps.append((pd.to_numeric(df[cols[0]],errors="coerce").to_numpy(float)+
                              pd.to_numeric(df[cols[1]],errors="coerce").to_numpy(float))/2.0)
        if len(comps)<2:
            return None,None
        norm=[]
        for s in comps:
            s=pd.Series(s).interpolate(limit_direction="both").rolling(
                max(3,int(round(fps*0.10))),center=True,min_periods=1).mean().to_numpy(float)
            med=np.nanmedian(s); scale=np.nanmedian(np.abs(s-med))*1.4826
            if not np.isfinite(scale) or scale<1e-8: scale=np.nanstd(s)+1e-8
            norm.append((s-med)/scale)
        sig=np.nanmean(np.vstack(norm),axis=0)
        # Eliminar componente muy lenta y enfatizar patrón de marcha.
        slow=pd.Series(sig).rolling(max(5,int(round(fps*0.70))),center=True,min_periods=1).mean().to_numpy()
        sig=sig-slow
        sig=(sig-np.nanmean(sig))/(np.nanstd(sig)+1e-8)
        return t,sig
    except Exception:
        return None,None

def estimate_v38_master_clock(front_snapshot,lateral_snapshot,coarse_offset_s,max_adjust_s=0.35,resample_hz=60.0):
    """
    Reloj físico maestro V38.
    Parte del offset global y sólo lo refina con una señal corporal continua;
    IC (Initial Contact / Contacto Inicial) no interviene en esta estimación.
    """
    tf,sf=_v38_master_clock_signal(front_snapshot)
    tl,sl=_v38_master_clock_signal(lateral_snapshot)
    coarse=float(coarse_offset_s)
    if tf is None or tl is None:
        return coarse,0.0,np.nan,"No calculable"
    best=None
    for adj in np.arange(-float(max_adjust_s),float(max_adjust_s)+1e-9,1.0/float(resample_hz)):
        off=coarse+float(adj)
        lo=max(float(tf[0]),float(tl[0])-off)
        hi=min(float(tf[-1]),float(tl[-1])-off)
        if hi-lo<0.8: continue
        grid=np.arange(lo,hi,1.0/float(resample_hz))
        if len(grid)<40: continue
        a=np.interp(grid,tf,sf,left=np.nan,right=np.nan)
        b=np.interp(grid+off,tl,sl,left=np.nan,right=np.nan)
        ok=np.isfinite(a)&np.isfinite(b)
        if ok.sum()<40: continue
        c=float(np.corrcoef(a[ok],b[ok])[0,1])
        if not np.isfinite(c): continue
        # Penalización suave para evitar grandes desplazamientos sin mejora clara.
        score=c-0.08*abs(adj)/max(float(max_adjust_s),1e-6)
        if best is None or score>best[0]:
            best=(score,off,float(adj),c)
    if best is None:
        return coarse,0.0,np.nan,"No calculable"
    _,off,adj,corr=best
    q="Alta" if corr>=0.65 else ("Moderada" if corr>=0.40 else "Baja")
    return float(off),float(adj),float(corr),q


def _v39_snapshot_ic_events(snapshot, prefer_audit=False):
    """Eventos IC aceptados, incluyendo el IC terminal de cada ciclo."""
    if not snapshot:
        return []
    fps=float(snapshot.get("fps",0) or 0)
    if fps<=0:
        return []
    out=[]
    info=snapshot.get("event_info") or {}
    if prefer_audit and info.get("accepted_audit"):
        for ev in info.get("accepted_audit",[]):
            fr=ev.get("frame",np.nan)
            if np.isfinite(fr):
                out.append({
                    "side":str(ev.get("side","?")),
                    "frame":int(round(float(fr))),
                    "time_s":float(fr)/fps,
                    "score":float(ev.get("score",np.nan)),
                    "source":"accepted_audit"
                })
    if not out:
        for side,obj in (("L",snapshot.get("left")),("R",snapshot.get("right"))):
            if not obj: continue
            vals=[]
            vals.extend(list(np.asarray(obj.get("ic",[]),dtype=float)))
            vals.extend(list(np.asarray(obj.get("next_ic",[]),dtype=float)))
            for fr in vals:
                if np.isfinite(fr):
                    out.append({"side":side,"frame":int(round(float(fr))),
                                "time_s":float(fr)/fps,"score":np.nan,"source":"cycle"})
    # deduplicar side+frame
    ded={}
    for ev in out:
        ded[(ev["side"],ev["frame"])]=ev
    return sorted(ded.values(),key=lambda z:z["time_s"])

def _v39_step_period(events):
    """Periodo de paso robusto a partir de alternancia L/R; fallback = zancada/2."""
    if not events or len(events)<2:
        return np.nan,0
    e=sorted(events,key=lambda z:z["time_s"])
    alt=[]
    same=[]
    for a,b in zip(e[:-1],e[1:]):
        dt=float(b["time_s"]-a["time_s"])
        if not (0.20<=dt<=1.30): continue
        if a["side"]!=b["side"]: alt.append(dt)
        else: same.append(dt/2.0)
    vals=alt if alt else same
    if not vals:
        return np.nan,0
    med=float(np.median(vals))
    good=[x for x in vals if abs(x-med)<=max(0.10,0.25*med)]
    if good: med=float(np.median(good))
    return med,len(good) if good else len(vals)

def _v39_phase_sync(front_snapshot,lateral_snapshot,anchor_offset_s,max_anchor_deviation_s=0.70):
    """
    Sincronización por fase periódica V39.
    Usa IC aceptados únicamente como marcadores de fase periódica, no como ajuste
    de un evento aislado. Exige al menos dos eventos laterales coherentes.
    """
    F=_v39_snapshot_ic_events(front_snapshot,prefer_audit=False)
    L=_v39_snapshot_ic_events(lateral_snapshot,prefer_audit=True)
    pf,nf=_v39_step_period(F); pl,nl=_v39_step_period(L)
    anchor=float(anchor_offset_s)
    base={
        "front_step_period_s":pf,"lateral_step_period_s":pl,
        "period_difference_pct":np.nan,"phase_offset_s":anchor,
        "phase_correction_from_master_s":0.0,"phase_quality":"No calculable",
        "phase_pairs":0,"phase_mae_s":np.nan,"phase_sequence_error_s":np.nan,
        "phase_lr_hypothesis":"No calculable","front_phase_events":F,"lateral_phase_events":L,
    }
    if not np.isfinite(pf) or not np.isfinite(pl) or len(F)<3 or len(L)<2:
        return base
    pdiff=abs(pl-pf)/max((pl+pf)/2.0,1e-6)*100.0
    base["period_difference_pct"]=float(pdiff)
    # Si los periodos difieren demasiado, no usar fase para recalibrar.
    if pdiff>18.0:
        base["phase_quality"]="No fiable"
        return base

    meanp=float((pf+pl)/2.0)
    tol=min(0.16,max(0.075,0.24*meanp))
    candidates=[anchor]
    # Offsets derivados de TODOS los eventos compatibles; un candidato sólo será
    # aceptado si consigue una secuencia de >=2 eventos.
    for swapped in (False,True):
        for f in F:
            for l in L:
                ls=("R" if l["side"]=="L" else "L") if swapped else l["side"]
                if f["side"]!=ls: continue
                off=float(l["time_s"]-f["time_s"])
                # Ambigüedad periódica: aceptar sólo soluciones próximas al reloj V38.
                if abs(off-anchor)<=float(max_anchor_deviation_s):
                    candidates.append(off)
    # Añadir versiones equivalentes +/- un periodo alrededor del ancla.
    expanded=[]
    for c in candidates:
        for k in (-1,0,1):
            cc=float(c+k*meanp)
            if abs(cc-anchor)<=float(max_anchor_deviation_s):
                expanded.append(cc)
    candidates=sorted(set(round(x,6) for x in expanded))

    def solve(off,swapped):
        C=[]
        for i,f in enumerate(F):
            for j,l in enumerate(L):
                ls=("R" if l["side"]=="L" else "L") if swapped else l["side"]
                if f["side"]!=ls: continue
                dt=(float(l["time_s"])-float(off))-float(f["time_s"])
                if abs(dt)<=tol:
                    C.append((i,j,float(dt)))
        C.sort(key=lambda q:(q[0],q[1]))
        best=[]
        def obj(ps):
            if not ps: return (0,-99,-99,-99)
            errs=np.asarray([abs(x[2]) for x in ps],float)
            signed=np.asarray([x[2] for x in ps],float)
            seqerr=0.0
            if len(ps)>=2:
                # Coherencia del intervalo entre pares, independiente del offset.
                ds=[]
                for x,y in zip(ps[:-1],ps[1:]):
                    fdt=F[y[0]]["time_s"]-F[x[0]]["time_s"]
                    ldt=L[y[1]]["time_s"]-L[x[1]]["time_s"]
                    ds.append(abs(ldt-fdt))
                seqerr=float(np.mean(ds)) if ds else 0.0
            else:
                seqerr=meanp
            return (len(ps),-seqerr,-float(np.mean(errs)),-float(np.std(signed)) if len(ps)>=2 else -9)
        def dfs(k,li,lj,chosen):
            nonlocal best
            if k>=len(C):
                if obj(chosen)>obj(best): best=list(chosen)
                return
            dfs(k+1,li,lj,chosen)
            i,j,dt=C[k]
            if i>li and j>lj:
                chosen.append(C[k]); dfs(k+1,i,j,chosen); chosen.pop()
        if len(C)<=24:
            dfs(0,-1,-1,[])
        else:
            ui=set(); uj=set()
            for q in sorted(C,key=lambda x:abs(x[2])):
                if q[0] not in ui and q[1] not in uj:
                    best.append(q); ui.add(q[0]); uj.add(q[1])
            best.sort()
        mae=float(np.mean([abs(x[2]) for x in best])) if best else np.nan
        seq=[]
        if len(best)>=2:
            for x,y in zip(best[:-1],best[1:]):
                seq.append(abs((L[y[1]]["time_s"]-L[x[1]]["time_s"])-
                               (F[y[0]]["time_s"]-F[x[0]]["time_s"])))
        seqerr=float(np.mean(seq)) if seq else np.nan
        return best,mae,seqerr

    ranked=[]
    for off in candidates:
        for swapped in (False,True):
            ps,mae,seqerr=solve(off,swapped)
            # ≥2 pares es condición para que una solución de fase pueda mover el reloj.
            support=len(ps)
            seqpen=seqerr if np.isfinite(seqerr) else 9.0
            maepen=mae if np.isfinite(mae) else 9.0
            # La proximidad al reloj maestro rompe empates y evita alias periódicos.
            rank=(support,-seqpen,-maepen,-abs(off-anchor))
            ranked.append((rank,off,swapped,ps,mae,seqerr))
    ranked.sort(key=lambda z:z[0],reverse=True)
    if not ranked:
        return base
    _,off,swapped,pairs,mae,seqerr=ranked[0]
    # Anti-espurio: no recalibrar con <2 pares; para 2 pares exigir intervalo coherente.
    if len(pairs)<2:
        base["phase_quality"]="Baja"
        return base
    if np.isfinite(seqerr) and seqerr>max(0.10,0.20*meanp):
        base["phase_quality"]="Baja"
        return base

    correction=float(off-anchor)
    if len(pairs)>=4 and mae<=0.06 and pdiff<=6:
        q="Alta"
    elif len(pairs)>=2 and mae<=0.12 and pdiff<=10:
        q="Moderada"
    else:
        q="Baja"
    base.update({
        "phase_offset_s":float(off),
        "phase_correction_from_master_s":correction,
        "phase_quality":q,
        "phase_pairs":int(len(pairs)),
        "phase_mae_s":float(mae) if np.isfinite(mae) else np.nan,
        "phase_sequence_error_s":float(seqerr) if np.isfinite(seqerr) else np.nan,
        "phase_lr_hypothesis":"L↔R invertido" if swapped else "L↔L / R↔R directo",
    })
    return base


def _v40_temporal_resolution(front_snapshot, lateral_snapshot):
    """Resolución e incertidumbre temporal según FPS (Frames Per Second / Fotogramas por Segundo)."""
    ffps=float((front_snapshot or {}).get("fps",0) or 0)
    lfps=float((lateral_snapshot or {}).get("fps",0) or 0)
    fframe=1.0/ffps if ffps>0 else np.nan
    lframe=1.0/lfps if lfps>0 else np.nan
    # ± medio frame por cámara; suma conservadora para diferencia entre dos marcas temporales.
    unc=(0.5*fframe if np.isfinite(fframe) else 0.0)+(0.5*lframe if np.isfinite(lframe) else 0.0)
    practical=max([x for x in (fframe,lframe) if np.isfinite(x)],default=np.nan)
    return ffps,lfps,fframe,lframe,float(unc),float(practical) if np.isfinite(practical) else np.nan

def _v40_match_event_sequence(front_snapshot,lateral_snapshot,offset_s,phase_diag=None):
    """
    V40 · empareja primero la secuencia de pasos y después construye ciclos:
    IC ipsilateral inicial -> IC contralateral -> siguiente IC ipsilateral.

    No exige que los dos detectores hayan construido previamente el mismo objeto cycle.
    Las tolerancias se adaptan a los FPS reales de cada cámara.
    """
    F=_v39_snapshot_ic_events(front_snapshot,prefer_audit=False)
    L=_v39_snapshot_ic_events(lateral_snapshot,prefer_audit=True)
    ffps,lfps,fframe,lframe,unc,practical=_v40_temporal_resolution(front_snapshot,lateral_snapshot)
    if len(F)<3 or len(L)<3:
        return [],[],{"event_pairs":0,"triplet_cycles":0,"second_lateral_homolog":0,
                     "event_mae_s":np.nan,"event_sequence_error_s":np.nan,
                     "lr_hypothesis":"No calculable","temporal_uncertainty_s":unc,
                     "practical_resolution_s":practical,"front_fps":ffps,"lateral_fps":lfps}

    # Umbral adaptativo: nunca inferior a 2 frames de la cámara más lenta.
    # Se limita para impedir coincidencias laxas.
    tol=max(0.075,2.0*practical if np.isfinite(practical) else 0.075)
    tol=min(tol,0.14)

    def solve(swapped):
        C=[]
        for i,f in enumerate(F):
            for j,l in enumerate(L):
                ls=("R" if l["side"]=="L" else "L") if swapped else l["side"]
                if f["side"]!=ls: continue
                dt=(float(l["time_s"])-float(offset_s))-float(f["time_s"])
                if abs(dt)<=tol:
                    C.append((i,j,float(dt)))
        C.sort(key=lambda q:(q[0],q[1]))
        best=[]
        def objective(ps):
            if not ps: return (0,-99,-99,-99)
            err=np.asarray([abs(x[2]) for x in ps],float)
            seq=[]
            for a,b in zip(ps[:-1],ps[1:]):
                seq.append(abs((F[b[0]]["time_s"]-F[a[0]]["time_s"])-
                               (L[b[1]]["time_s"]-L[a[1]]["time_s"])))
            se=float(np.mean(seq)) if seq else 9.0
            return (len(ps),-se,-float(np.mean(err)),-float(np.std([x[2] for x in ps])) if len(ps)>=2 else -9)
        def dfs(k,li,lj,chosen):
            nonlocal best
            if k>=len(C):
                if objective(chosen)>objective(best): best=list(chosen)
                return
            dfs(k+1,li,lj,chosen)
            i,j,dt=C[k]
            if i>li and j>lj:
                chosen.append(C[k]); dfs(k+1,i,j,chosen); chosen.pop()
        if len(C)<=26:
            dfs(0,-1,-1,[])
        else:
            ui=set();uj=set()
            for q in sorted(C,key=lambda x:abs(x[2])):
                if q[0] not in ui and q[1] not in uj:
                    best.append(q);ui.add(q[0]);uj.add(q[1])
            best.sort()
        return best

    direct=solve(False); swapped=solve(True)
    def rank(ps):
        if not ps:return (0,-99,-99)
        e=np.mean([abs(x[2]) for x in ps])
        seq=[]
        for a,b in zip(ps[:-1],ps[1:]):
            seq.append(abs((F[b[0]]["time_s"]-F[a[0]]["time_s"])-
                           (L[b[1]]["time_s"]-L[a[1]]["time_s"])))
        return (len(ps),-(np.mean(seq) if seq else 9),-e)
    use_swapped=rank(swapped)>rank(direct)
    chosen=swapped if use_swapped else direct

    event_pairs=[]
    for i,j,dt in chosen:
        event_pairs.append({
            "front":F[i],"lateral":L[j],"delta_s":float(dt),
            "lr_swapped":bool(use_swapped),
            "delta_front_frames":float(abs(dt)*ffps) if ffps>0 else np.nan,
            "delta_lateral_frames":float(abs(dt)*lfps) if lfps>0 else np.nan,
        })

    # Construcción de tripletes consecutivos A-B-A en la secuencia ya emparejada.
    triplets=[]
    for k in range(len(event_pairs)-2):
        p0,p1,p2=event_pairs[k:k+3]
        sf=[p0["front"]["side"],p1["front"]["side"],p2["front"]["side"]]
        sl=[p0["lateral"]["side"],p1["lateral"]["side"],p2["lateral"]["side"]]
        if not (sf[0]==sf[2] and sf[0]!=sf[1]): continue
        # En lateral se exige también alternancia ipsi-contra-ipsi, antes o después de inversión.
        mapped=[("R" if x=="L" else "L") if use_swapped else x for x in sl]
        if not (mapped[0]==mapped[2] and mapped[0]!=mapped[1]): continue
        f_dur=float(p2["front"]["time_s"]-p0["front"]["time_s"])
        l_dur=float(p2["lateral"]["time_s"]-p0["lateral"]["time_s"])
        dur_diff=abs(l_dur-f_dur)
        mid_err=abs(float(p1["delta_s"]))
        # Duración tolerada en función de cuantización: mínimo 2 frames laterales, máximo 120 ms.
        dur_tol=min(0.12,max(0.055,2.0*practical if np.isfinite(practical) else 0.055))
        if dur_diff>dur_tol: continue
        triplets.append({
            "side":sf[0],"front_start":p0["front"],"front_mid":p1["front"],"front_end":p2["front"],
            "lateral_start":p0["lateral"],"lateral_mid":p1["lateral"],"lateral_end":p2["lateral"],
            "start_delta_s":float(p0["delta_s"]),"mid_delta_s":float(p1["delta_s"]),
            "end_delta_s":float(p2["delta_s"]),"duration_delta_s":float(dur_diff),
            "front_duration_s":f_dur,"lateral_duration_s":l_dur,
            "lr_swapped":bool(use_swapped)
        })

    errs=[abs(p["delta_s"]) for p in event_pairs]
    seqerrs=[]
    for a,b in zip(event_pairs[:-1],event_pairs[1:]):
        seqerrs.append(abs((b["front"]["time_s"]-a["front"]["time_s"])-
                           (b["lateral"]["time_s"]-a["lateral"]["time_s"])))
    lateral_cycle_starts={(t["lateral_start"]["side"],t["lateral_start"]["frame"]) for t in triplets}
    diag={
        "event_pairs":len(event_pairs),"triplet_cycles":len(triplets),
        "second_lateral_homolog":1 if len(lateral_cycle_starts)>=2 else 0,
        "event_mae_s":float(np.mean(errs)) if errs else np.nan,
        "event_sequence_error_s":float(np.mean(seqerrs)) if seqerrs else np.nan,
        "lr_hypothesis":"L↔R invertido" if use_swapped else "L↔L / R↔R directo",
        "direct_event_pairs":len(direct),"swapped_event_pairs":len(swapped),
        "temporal_uncertainty_s":unc,"practical_resolution_s":practical,
        "front_frame_s":fframe,"lateral_frame_s":lframe,"front_fps":ffps,"lateral_fps":lfps,
        "event_tolerance_s":tol,
    }
    return event_pairs,triplets,diag


def _v41_periodic_triplet_rescue(front_snapshot,lateral_snapshot,offset_s,v40_event_pairs=None,phase_diag=None):
    """
    V41 · recuperación conservadora del tercer IC (Initial Contact / Contacto Inicial).
    Parte de correspondencias V40 fuertes y busca un tercer IC REAL observado que:
      1) respete alternancia L/R,
      2) esté cerca del tiempo previsto por el periodo de paso,
      3) sea compatible en ambas cámaras,
      4) no requiera modificar el offset de sincronización.
    Nunca crea/inventa un IC ausente.
    """
    F=_v39_snapshot_ic_events(front_snapshot,prefer_audit=False)
    L=_v39_snapshot_ic_events(lateral_snapshot,prefer_audit=True)
    ffps,lfps,fframe,lframe,unc,practical=_v40_temporal_resolution(front_snapshot,lateral_snapshot)
    strong=list(v40_event_pairs or [])
    fd=float((phase_diag or {}).get("front_step_period_s",np.nan))
    ld=float((phase_diag or {}).get("lateral_step_period_s",np.nan))
    periods=[x for x in (fd,ld) if np.isfinite(x) and 0.20<=x<=1.30]
    step=float(np.median(periods)) if periods else np.nan
    diag={"attempted":0,"rescued_events":0,"triplet_cycles":0,"second_lateral_homolog":0,
          "predicted_step_period_s":step,"rescue_tolerance_s":np.nan,
          "rescue_rows":[],"triplets":[],"v42_triplet_audit":[],"v42_candidate_audit":[],
          "v43_closure_cycles":[],"v43_closure_cycles_count":0,"v43_second_lateral_homolog":0}
    if len(strong)<2 or not np.isfinite(step):
        return diag

    # V41 NO ensancha el emparejamiento general.
    # Solo permite rescate alrededor del instante predicho: 2.5 frames de la cámara más lenta,
    # con suelo 85 ms y techo 115 ms.
    rescue_tol=min(0.115,max(0.085,2.5*practical if np.isfinite(practical) else 0.085))
    diag["rescue_tolerance_s"]=rescue_tol
    diag["attempted"]=1

    # Inferir hipótesis anatómica desde los pares fuertes.
    swapped=bool(strong[0].get("lr_swapped",False))
    usedF={(p["front"]["side"],p["front"]["frame"]) for p in strong}
    usedL={(p["lateral"]["side"],p["lateral"]["frame"]) for p in strong}
    candidates=[]

    # Extender desde cada extremo fuerte hacia delante y atrás un paso.
    for base in strong:
        for direction in (-1,1):
            tf_pred=float(base["front"]["time_s"])+direction*step
            tl_pred=float(base["lateral"]["time_s"])+direction*step
            expected_f="R" if base["front"]["side"]=="L" else "L"
            expected_l="R" if base["lateral"]["side"]=="L" else "L"
            for fi,f in enumerate(F):
                if f["side"]!=expected_f or (f["side"],f["frame"]) in usedF: continue
                ef=abs(float(f["time_s"])-tf_pred)
                if ef>rescue_tol: continue
                for li,l in enumerate(L):
                    if l["side"]!=expected_l or (l["side"],l["frame"]) in usedL: continue
                    el=abs(float(l["time_s"])-tl_pred)
                    if el>rescue_tol: continue
                    # Debe seguir siendo compatible frontal-lateral con el reloj ya fijado.
                    dt=(float(l["time_s"])-float(offset_s))-float(f["time_s"])
                    cross_tol=min(0.115,max(0.085,3.0*practical if np.isfinite(practical) else 0.085))
                    _cand_status="compatible" if abs(dt)<=cross_tol else "rechazado_delta_biplanar"
                    diag["v42_candidate_audit"].append({
                        "front_side":f["side"],"front_frame":f["frame"],"front_time_s":float(f["time_s"]),
                        "lateral_side":l["side"],"lateral_frame":l["frame"],"lateral_time_s":float(l["time_s"]),
                        "predicted_front_s":float(tf_pred),"predicted_lateral_s":float(tl_pred),
                        "prediction_error_front_s":float(ef),"prediction_error_lateral_s":float(el),
                        "cross_delta_s":float(dt),"cross_tolerance_s":float(cross_tol),
                        "status":_cand_status,"direction":"siguiente" if direction>0 else "anterior"
                    })
                    if abs(dt)>cross_tol: continue
                    # Penaliza desviación de predicción y discordancia entre cámaras.
                    score=ef+el+0.75*abs(dt)
                    candidates.append((score,fi,li,dt,ef,el,direction,base))

    # Selección conservadora uno-a-uno.
    rescued=[]
    for score,fi,li,dt,ef,el,direction,base in sorted(candidates,key=lambda x:x[0]):
        f,l=F[fi],L[li]
        if (f["side"],f["frame"]) in usedF or (l["side"],l["frame"]) in usedL: continue
        p={"front":f,"lateral":l,"delta_s":float(dt),"lr_swapped":swapped,
           "delta_front_frames":abs(float(dt))*ffps if ffps>0 else np.nan,
           "delta_lateral_frames":abs(float(dt))*lfps if lfps>0 else np.nan,
           "v41_rescued":True,"prediction_error_front_s":float(ef),
           "prediction_error_lateral_s":float(el)}
        rescued.append(p); usedF.add((f["side"],f["frame"])); usedL.add((l["side"],l["frame"]))
        diag["rescue_rows"].append({
            "front_side":f["side"],"front_frame":f["frame"],"front_time_s":float(f["time_s"]),
            "lateral_side":l["side"],"lateral_frame":l["frame"],"lateral_time_s":float(l["time_s"]),
            "cross_delta_s":float(dt),"prediction_error_front_s":float(ef),
            "prediction_error_lateral_s":float(el)
        })

    allpairs=strong+rescued
    allpairs.sort(key=lambda p:float(p["front"]["time_s"]))
    diag["rescued_events"]=len(rescued)

    trip=[]
    for k in range(len(allpairs)-2):
        a,b,c=allpairs[k:k+3]
        sf=[a["front"]["side"],b["front"]["side"],c["front"]["side"]]
        sl=[a["lateral"]["side"],b["lateral"]["side"],c["lateral"]["side"]]
        mapped=[("R" if x=="L" else "L") if swapped else x for x in sl]
        front_alt=bool(sf[0]==sf[2] and sf[0]!=sf[1])
        lateral_alt=bool(mapped[0]==mapped[2] and mapped[0]!=mapped[1])
        f1=float(b["front"]["time_s"]-a["front"]["time_s"]); f2=float(c["front"]["time_s"]-b["front"]["time_s"])
        l1=float(b["lateral"]["time_s"]-a["lateral"]["time_s"]); l2=float(c["lateral"]["time_s"]-b["lateral"]["time_s"])
        per_errors=[abs(f1-step),abs(f2-step),abs(l1-step),abs(l2-step)]
        periodic_ok=bool(max(per_errors)<=rescue_tol)
        fdur=float(c["front"]["time_s"]-a["front"]["time_s"])
        ldur=float(c["lateral"]["time_s"]-a["lateral"]["time_s"])
        duration_tol=max(0.07,2.5*practical if np.isfinite(practical) else 0.07)
        duration_delta=abs(fdur-ldur)
        duration_ok=bool(duration_delta<=duration_tol)

        # V43 · validación por cierre de ciclo.
        # No corrige ni desplaza el IC intermedio: determina si IC1 e IC3 delimitan
        # el mismo ciclo físico en ambas cámaras y etiqueta por separado la discrepancia interna.
        endpoint_tol=min(0.10,max(0.060,2.0*practical if np.isfinite(practical) else 0.060))
        start_cross=abs(float(a.get("delta_s",np.nan)))
        mid_cross=abs(float(b.get("delta_s",np.nan)))
        end_cross=abs(float(c.get("delta_s",np.nan)))
        endpoints_ok=bool(np.isfinite(start_cross) and np.isfinite(end_cross) and
                          start_cross<=endpoint_tol and end_cross<=endpoint_tol)
        lateral_periodicity_ok=bool(abs(l1-step)<=rescue_tol and abs(l2-step)<=rescue_tol)
        # Cierre homólogo: alternancia anatómica + extremos coincidentes + duración global
        # + periodicidad lateral. El IC medio frontal puede quedar marcado como discrepante.
        closure_ok=bool(front_alt and lateral_alt and endpoints_ok and duration_ok and lateral_periodicity_ok)
        middle_localized_discrepancy=bool(
            closure_ok and (
                abs(f1-step)>rescue_tol or abs(f2-step)>rescue_tol or mid_cross>endpoint_tol
            )
        )
        closure_class=(
            "Homólogo con discrepancia localizada del IC intermedio frontal"
            if middle_localized_discrepancy else
            ("Homólogo por cierre de ciclo" if closure_ok else "No homólogo por cierre")
        )
        reasons=[]
        if not front_alt: reasons.append("alternancia frontal no cumple A-B-A")
        if not lateral_alt: reasons.append("alternancia lateral mapeada no cumple A-B-A")
        if not periodic_ok:
            labels=["frontal IC1→IC2","frontal IC2→IC3","lateral IC1→IC2","lateral IC2→IC3"]
            worst=int(np.argmax(per_errors))
            reasons.append(f"periodicidad fuera de ventana: {labels[worst]} error={1000*per_errors[worst]:.1f} ms > {1000*rescue_tol:.1f} ms")
        if not duration_ok:
            reasons.append(f"duración frontal-lateral difiere {1000*duration_delta:.1f} ms > {1000*duration_tol:.1f} ms")
        diag["v42_triplet_audit"].append({
            "triplet_index":k+1,
            "front_sides":"-".join(sf),"lateral_sides":"-".join(sl),
            "mapped_lateral_sides":"-".join(mapped),
            "front_frames":f'{a["front"]["frame"]}-{b["front"]["frame"]}-{c["front"]["frame"]}',
            "lateral_frames":f'{a["lateral"]["frame"]}-{b["lateral"]["frame"]}-{c["lateral"]["frame"]}',
            "front_times_s":f'{float(a["front"]["time_s"]):.6f}-{float(b["front"]["time_s"]):.6f}-{float(c["front"]["time_s"]):.6f}',
            "lateral_times_s":f'{float(a["lateral"]["time_s"]):.6f}-{float(b["lateral"]["time_s"]):.6f}-{float(c["lateral"]["time_s"]):.6f}',
            "front_interval_1_ms":1000*f1,"front_interval_2_ms":1000*f2,
            "lateral_interval_1_ms":1000*l1,"lateral_interval_2_ms":1000*l2,
            "expected_step_ms":1000*step,
            "front_error_1_ms":1000*per_errors[0],"front_error_2_ms":1000*per_errors[1],
            "lateral_error_1_ms":1000*per_errors[2],"lateral_error_2_ms":1000*per_errors[3],
            "duration_front_ms":1000*fdur,"duration_lateral_ms":1000*ldur,
            "duration_delta_ms":1000*duration_delta,"duration_tolerance_ms":1000*duration_tol,
            "front_alternation_ok":front_alt,"lateral_alternation_ok":lateral_alt,
            "periodicity_ok":periodic_ok,"duration_ok":duration_ok,
            "v43_endpoint_tolerance_ms":1000*endpoint_tol,
            "v43_start_cross_delta_ms":1000*start_cross if np.isfinite(start_cross) else np.nan,
            "v43_mid_cross_delta_ms":1000*mid_cross if np.isfinite(mid_cross) else np.nan,
            "v43_end_cross_delta_ms":1000*end_cross if np.isfinite(end_cross) else np.nan,
            "v43_endpoints_ok":endpoints_ok,
            "v43_lateral_periodicity_ok":lateral_periodicity_ok,
            "v43_closure_ok":closure_ok,
            "v43_middle_ic_discrepancy":middle_localized_discrepancy,
            "v43_closure_class":closure_class,
            "accepted":bool(front_alt and lateral_alt and periodic_ok and duration_ok),
            "rejection_reason":"ACEPTADO" if not reasons else " | ".join(reasons)
        })
        if closure_ok:
            diag["v43_closure_cycles"].append({
                "side":sf[0],
                "front_start_frame":a["front"]["frame"],"front_mid_frame":b["front"]["frame"],"front_end_frame":c["front"]["frame"],
                "lateral_start_frame":a["lateral"]["frame"],"lateral_mid_frame":b["lateral"]["frame"],"lateral_end_frame":c["lateral"]["frame"],
                "front_duration_s":fdur,"lateral_duration_s":ldur,"duration_delta_s":duration_delta,
                "start_delta_s":float(a.get("delta_s",np.nan)),"mid_delta_s":float(b.get("delta_s",np.nan)),
                "end_delta_s":float(c.get("delta_s",np.nan)),
                "middle_ic_discrepancy":middle_localized_discrepancy,
                "classification":closure_class,
                "lr_swapped":swapped
            })
        if reasons: continue
        trip.append({
            "side":sf[0],"front_start":a["front"],"front_mid":b["front"],"front_end":c["front"],
            "lateral_start":a["lateral"],"lateral_mid":b["lateral"],"lateral_end":c["lateral"],
            "front_duration_s":fdur,"lateral_duration_s":ldur,
            "duration_delta_s":abs(fdur-ldur),"start_delta_s":a["delta_s"],
            "mid_delta_s":b["delta_s"],"end_delta_s":c["delta_s"],
            "lr_swapped":swapped,"contains_rescued_event":bool(a.get("v41_rescued") or b.get("v41_rescued") or c.get("v41_rescued"))
        })
    diag["triplets"]=trip
    diag["triplet_cycles"]=len(trip)
    latstarts={(t["lateral_start"]["side"],t["lateral_start"]["frame"]) for t in trip}
    diag["second_lateral_homolog"]=1 if len(latstarts)>=2 else 0
    diag["v43_closure_cycles_count"]=len(diag.get("v43_closure_cycles",[]))
    _v43starts={(t.get("lateral_start_frame"),t.get("side")) for t in diag.get("v43_closure_cycles",[])}
    diag["v43_second_lateral_homolog"]=1 if len(_v43starts)>=2 else 0
    return diag


def _v44_chain_consecutive_cycles(front_snapshot,lateral_snapshot,offset_s,phase_diag=None):
    """
    V44 · encadenamiento de ciclos biplanares consecutivos.
    Trabaja sobre IC (Initial Contact / Contacto Inicial) realmente detectados.
    Busca secuencias alternantes de 5 eventos A-B-A-B-A que contienen dos ciclos
    consecutivos y obliga a que el IC final del ciclo 1 sea exactamente el IC inicial
    del ciclo 2 en cada cámara. No reajusta el reloj ni crea eventos.
    """
    F=_v39_snapshot_ic_events(front_snapshot,prefer_audit=False)
    L=_v39_snapshot_ic_events(lateral_snapshot,prefer_audit=True)
    ffps,lfps,fframe,lframe,unc,practical=_v40_temporal_resolution(front_snapshot,lateral_snapshot)
    fd=float((phase_diag or {}).get("front_step_period_s",np.nan))
    ld=float((phase_diag or {}).get("lateral_step_period_s",np.nan))
    periods=[x for x in (fd,ld) if np.isfinite(x) and 0.20<=x<=1.30]
    step=float(np.median(periods)) if periods else np.nan
    endpoint_tol=min(0.10,max(0.060,2.0*practical if np.isfinite(practical) else 0.060))
    middle_tol=min(0.13,max(0.095,3.0*practical if np.isfinite(practical) else 0.095))
    duration_tol=max(0.07,2.5*practical if np.isfinite(practical) else 0.07)
    out={"chains":[],"chain_count":0,"cycles_count":0,"second_cycle_homolog":0,
         "step_period_s":step,"endpoint_tolerance_s":endpoint_tol,
         "middle_tolerance_s":middle_tol,"duration_tolerance_s":duration_tol,
         "direct_chains":0,"swapped_chains":0,"audit":[]}
    if len(F)<5 or len(L)<5 or not np.isfinite(step):
        return out

    def mapped_side(side,sw):
        return ("R" if side=="L" else "L") if sw else side

    def alternating(seq):
        ss=[x["side"] for x in seq]
        return len(ss)==5 and all(ss[i]!=ss[i+1] for i in range(4)) and ss[0]==ss[2]==ss[4] and ss[1]==ss[3]

    # Consecutive windows of 5 accepted events only; no skipping inside a chain.
    FW=[F[i:i+5] for i in range(len(F)-4) if alternating(F[i:i+5])]
    LW=[L[i:i+5] for i in range(len(L)-4) if alternating(L[i:i+5])]
    candidates=[]
    for sw in (False,True):
        for fi,fs in enumerate(FW):
            for li,ls in enumerate(LW):
                if any(fs[k]["side"]!=mapped_side(ls[k]["side"],sw) for k in range(5)):
                    continue
                # Physical lateral times after fixed offset.
                ft=np.asarray([float(x["time_s"]) for x in fs],float)
                lt=np.asarray([float(x["time_s"])-float(offset_s) for x in ls],float)
                delta=lt-ft
                # IC 0,2,4 are boundaries of the two full cycles and must be tight.
                boundary_ok=bool(np.all(np.abs(delta[[0,2,4]])<=endpoint_tol))
                # IC 1,3 may expose a localized detector discrepancy but remain bounded.
                middle_ok=bool(np.all(np.abs(delta[[1,3]])<=middle_tol))
                fints=np.diff(ft); lints=np.diff(lt)
                # Lateral periodicity remains the stable reference.
                lat_period_ok=bool(np.all(np.abs(lints-step)<=max(0.085,2.75*practical)))
                # Two cycle durations: 0→2 and 2→4.
                fdurs=np.asarray([ft[2]-ft[0],ft[4]-ft[2]],float)
                ldurs=np.asarray([lt[2]-lt[0],lt[4]-lt[2]],float)
                dd=np.abs(fdurs-ldurs)
                duration_ok=bool(np.all(dd<=duration_tol))
                # Continuity is structural: same exact event index 2 closes C1 and opens C2.
                continuity_ok=True
                accepted=bool(boundary_ok and middle_ok and lat_period_ok and duration_ok and continuity_ok)
                reasons=[]
                if not boundary_ok: reasons.append("IC de frontera fuera de tolerancia")
                if not middle_ok: reasons.append("IC intermedio fuera del límite localizado")
                if not lat_period_ok: reasons.append("periodicidad lateral incoherente")
                if not duration_ok: reasons.append("duración de uno de los ciclos incoherente")
                row={
                    "hypothesis":"L/R invertido" if sw else "directo",
                    "front_frames":"-".join(str(x["frame"]) for x in fs),
                    "lateral_frames":"-".join(str(x["frame"]) for x in ls),
                    "delta_ic1_ms":1000*delta[0],"delta_ic2_ms":1000*delta[1],
                    "delta_ic3_shared_ms":1000*delta[2],"delta_ic4_ms":1000*delta[3],
                    "delta_ic5_ms":1000*delta[4],
                    "cycle1_front_ms":1000*fdurs[0],"cycle1_lateral_ms":1000*ldurs[0],
                    "cycle1_duration_delta_ms":1000*dd[0],
                    "cycle2_front_ms":1000*fdurs[1],"cycle2_lateral_ms":1000*ldurs[1],
                    "cycle2_duration_delta_ms":1000*dd[1],
                    "boundary_ok":boundary_ok,"middle_ok":middle_ok,
                    "lateral_periodicity_ok":lat_period_ok,"duration_ok":duration_ok,
                    "continuity_shared_ic":True,"accepted":accepted,
                    "reason":"ACEPTADO" if accepted else " | ".join(reasons)
                }
                out["audit"].append(row)
                if accepted:
                    score=float(np.mean(np.abs(delta[[0,2,4]]))+0.35*np.mean(np.abs(delta[[1,3]]))+0.5*np.mean(dd))
                    candidates.append((score,sw,fs,ls,delta,fdurs,ldurs,dd,row))

    if not candidates:
        return out
    # Conservative: select best chain only; prevents overlapping alternative hypotheses.
    score,sw,fs,ls,delta,fdurs,ldurs,dd,row=min(candidates,key=lambda x:x[0])
    chain={
        "hypothesis":"L/R invertido" if sw else "directo",
        "front_frames":[int(x["frame"]) for x in fs],
        "lateral_frames":[int(x["frame"]) for x in ls],
        "front_sides":[x["side"] for x in fs],
        "lateral_sides":[x["side"] for x in ls],
        "delta_events_s":[float(x) for x in delta],
        "cycle1_front_s":float(fdurs[0]),"cycle1_lateral_s":float(ldurs[0]),
        "cycle1_duration_delta_s":float(dd[0]),
        "cycle2_front_s":float(fdurs[1]),"cycle2_lateral_s":float(ldurs[1]),
        "cycle2_duration_delta_s":float(dd[1]),
        "shared_front_frame":int(fs[2]["frame"]),
        "shared_lateral_frame":int(ls[2]["frame"]),
        "middle1_discrepancy":bool(abs(delta[1])>endpoint_tol),
        "middle2_discrepancy":bool(abs(delta[3])>endpoint_tol),
        "continuity_shared_ic":True,
        "score":score
    }
    out["chains"]=[chain]
    out["chain_count"]=1
    out["cycles_count"]=2
    out["second_cycle_homolog"]=1
    out["swapped_chains"]=1 if sw else 0
    out["direct_chains"]=0 if sw else 1
    return out


def _v45_second_cycle_evaluability(front_snapshot,lateral_snapshot,offset_s,phase_diag=None,v44_diag=None):
    """
    V45 · determina si el segundo ciclo lateral es realmente evaluable.
    Si no existe una cadena V44 de cinco IC, predice el siguiente IC a partir del
    último IC lateral aceptado y del periodo de paso robusto, y compara ese instante
    con el final de la ventana lateral visible.
    No crea ni rescata eventos y no modifica la sincronización.
    """
    L=_v39_snapshot_ic_events(lateral_snapshot,prefer_audit=True)
    fps=float((lateral_snapshot or {}).get("fps",0) or 0)
    w=(lateral_snapshot or {}).get("visibility_window") or {}
    end_frame=float(w.get("end_frame",np.nan))
    start_frame=float(w.get("start_frame",np.nan))
    phase_diag=phase_diag or {}
    v44_diag=v44_diag or {}
    pl=float(phase_diag.get("lateral_step_period_s",np.nan))
    pf=float(phase_diag.get("front_step_period_s",np.nan))
    periods=[x for x in (pl,pf) if np.isfinite(x) and 0.20<=x<=1.30]
    step=float(np.median(periods)) if periods else np.nan
    practical=float(v44_diag.get("endpoint_tolerance_s",np.nan))
    if not np.isfinite(practical):
        _,_,_,lframe,_,pr=_v40_temporal_resolution(front_snapshot,lateral_snapshot)
        practical=pr

    out={
        "status":"No evaluable","reason":"Datos insuficientes",
        "second_cycle_evaluable":0,"coverage_sufficient":0,
        "predicted_next_ic_camera_s":np.nan,"predicted_next_ic_frame":np.nan,
        "predicted_next_ic_physical_s":np.nan,"visibility_end_frame":end_frame,
        "visibility_end_camera_s":np.nan,"visibility_margin_s":np.nan,
        "last_accepted_side":"","last_accepted_frame":np.nan,
        "last_accepted_camera_s":np.nan,"expected_next_side":"",
        "step_period_s":step,"accepted_lateral_events":len(L),
        "rejected_near_prediction":[],
    }

    if fps<=0 or not L or not np.isfinite(step):
        out["reason"]="No hay FPS lateral, periodo de paso o IC aceptados suficientes."
        return out

    # Si V44 ya validó la segunda cadena, es evaluable y positiva.
    if int(v44_diag.get("second_cycle_homolog",0) or 0)>=1:
        out.update({
            "status":"Evaluable · segundo ciclo homólogo validado",
            "reason":"V44 validó una cadena A-B-A-B-A de dos ciclos consecutivos.",
            "second_cycle_evaluable":1,"coverage_sufficient":1
        })
        return out

    last=L[-1]
    t_last=float(last["time_s"])
    pred=t_last+step
    pred_frame=pred*fps
    pred_phys=pred-float(offset_s)
    expected="R" if str(last["side"])=="L" else "L"
    out.update({
        "last_accepted_side":str(last["side"]),
        "last_accepted_frame":int(last["frame"]),
        "last_accepted_camera_s":t_last,
        "expected_next_side":expected,
        "predicted_next_ic_camera_s":pred,
        "predicted_next_ic_frame":pred_frame,
        "predicted_next_ic_physical_s":pred_phys,
    })

    if np.isfinite(end_frame):
        end_s=end_frame/fps
        margin=end_s-pred
        out["visibility_end_camera_s"]=end_s
        out["visibility_margin_s"]=margin
        # Exigimos que el evento esperado tenga al menos medio frame de margen visible.
        half_frame=0.5/fps
        if pred > end_s-half_frame:
            out.update({
                "status":"No evaluable · cobertura temporal lateral insuficiente",
                "reason":"La ventana lateral termina antes (o prácticamente en el mismo instante) del siguiente IC esperado.",
                "second_cycle_evaluable":0,"coverage_sufficient":0
            })
        else:
            out.update({
                "status":"Evaluable · siguiente IC esperado dentro de la ventana lateral",
                "reason":"La cámara lateral mantiene cobertura temporal suficiente hasta el siguiente IC esperado, pero éste no quedó aceptado.",
                "second_cycle_evaluable":1,"coverage_sufficient":1
            })
    else:
        # Sin final de visibilidad fiable no convertir ausencia en evidencia negativa.
        out.update({
            "status":"No evaluable · final de visibilidad lateral no determinado",
            "reason":"No puede verificarse si el siguiente IC esperado cae dentro de la cobertura lateral.",
            "second_cycle_evaluable":0,"coverage_sufficient":0
        })

    # Auditoría informativa: candidatos rechazados próximos a la predicción.
    info=(lateral_snapshot or {}).get("event_info") or {}
    near=[]
    for ev in info.get("candidate_audit",[]) or []:
        fr=float(ev.get("frame",np.nan))
        if not np.isfinite(fr): continue
        dt=(fr/fps)-pred
        if abs(dt)<=max(0.20,0.40*step):
            near.append({
                "side":str(ev.get("side","?")),"frame":int(round(fr)),
                "status":str(ev.get("status","")),
                "score":float(ev.get("score",np.nan)),
                "reason":str(ev.get("reason","")),
                "delta_to_predicted_ms":1000*dt
            })
    out["rejected_near_prediction"]=near
    return out

def match_biplanar_cycles(front_snapshot, lateral_snapshot, sync_offset_s, tolerance_s=0.22, valid_front=None, valid_lateral=None, physical_window=None, physical_clock_override_s=None):
    """
    Versión 39 · sincronización por fase periódica de la marcha.

    1) Mantiene el reloj físico maestro V38 como referencia independiente.
    2) Estima periodo de paso frontal y lateral.
    3) Busca una correspondencia de FASE de la secuencia completa, no de un IC aislado.
    4) Sólo permite que la fase refine el offset si hay >=2 eventos laterales coherentes.
    5) Evalúa hipótesis L/R directa e invertida.
    """
    A_all=_cycles_from_snapshot(front_snapshot); B_all=_cycles_from_snapshot(lateral_snapshot)
    coarse=float(sync_offset_s)
    master_off,master_adj,master_corr,master_q=estimate_v38_master_clock(
        front_snapshot,lateral_snapshot,coarse
    )

    # V46 · separar el reloj físico de la alineación biomecánica.
    # Si existe una referencia temporal conocida (p. ej., cámaras sincronizadas por hardware
    # o un dataset con ground truth), ésta fija el reloj físico y NO puede ser movida por IC/TO
    # ni por la fase periódica de la marcha. En uso clínico normal, se conserva el reloj V38.
    _override_valid = False
    try:
        _override_valid = physical_clock_override_s is not None and np.isfinite(float(physical_clock_override_s))
    except Exception:
        _override_valid = False
    physical_clock_off = float(physical_clock_override_s) if _override_valid else float(master_off)
    physical_clock_source = "Referencia temporal conocida" if _override_valid else "Reloj maestro V38"

    # V47 · benchmark cuantitativo independiente del reloj.
    # En modo validación, el ground truth gobierna el emparejamiento, pero V38 sigue
    # estimándose de forma ciega y se compara CONTRA la referencia conocida. Así el
    # error ya no es 0 por construcción: mide la capacidad real del estimador interno.
    ffps47, lfps47, fframe47, lframe47, unc47, practical47 = _v40_temporal_resolution(front_snapshot, lateral_snapshot)
    v47_estimator_error = np.nan
    v47_abs_error = np.nan
    v47_error_frames = np.nan
    v47_class = "No aplicable"
    if _override_valid and np.isfinite(master_off):
        v47_estimator_error = float(master_off) - float(physical_clock_override_s)
        v47_abs_error = abs(v47_estimator_error)
        if np.isfinite(practical47) and practical47 > 0:
            v47_error_frames = v47_abs_error / practical47
            if v47_abs_error <= practical47:
                v47_class = "Dentro de 1 frame práctico"
            elif v47_abs_error <= 2.0 * practical47:
                v47_class = "Entre 1 y 2 frames prácticos"
            else:
                v47_class = "Fuera de 2 frames prácticos"
        else:
            v47_class = "Ground truth disponible · resolución temporal no calculable"

    # La fase biomecánica se estima respecto al reloj físico, pero queda registrada como
    # una magnitud distinta. Sólo puede gobernar el emparejamiento en modo clínico normal.
    phase=_v39_phase_sync(front_snapshot,lateral_snapshot,physical_clock_off)
    phase_q=str(phase.get("phase_quality","No calculable"))
    phase_off=float(phase.get("phase_offset_s",physical_clock_off))
    phase_supported=(int(phase.get("phase_pairs",0))>=2 and phase_q not in ("No fiable","No calculable"))
    match_off = physical_clock_off if _override_valid else (phase_off if phase_supported else physical_clock_off)

    diag={
        "coarse_offset_s":coarse,
        "fine_correction_s":master_adj,
        "final_offset_s":match_off,
        "master_clock_offset_s":master_off,
        "v46_physical_clock_offset_s":physical_clock_off,
        "v46_physical_clock_source":physical_clock_source,
        "v46_ground_truth_active":1 if _override_valid else 0,
        "v46_ground_truth_offset_s":float(physical_clock_override_s) if _override_valid else np.nan,
        "v46_physical_clock_error_s":(physical_clock_off-float(physical_clock_override_s)) if _override_valid else np.nan,
        "v46_biomechanical_phase_offset_s":phase_off,
        "v46_phase_minus_physical_s":phase_off-physical_clock_off,
        "v46_matching_offset_s":match_off,
        "v47_independent_clock_estimate_s":float(master_off) if np.isfinite(master_off) else np.nan,
        "v47_clock_estimator_error_s":v47_estimator_error,
        "v47_clock_estimator_abs_error_s":v47_abs_error,
        "v47_clock_estimator_error_practical_frames":v47_error_frames,
        "v47_validation_class":v47_class,
        "v47_front_frame_ms":1000.0*fframe47 if np.isfinite(fframe47) else np.nan,
        "v47_lateral_frame_ms":1000.0*lframe47 if np.isfinite(lframe47) else np.nan,
        "v47_temporal_uncertainty_ms":1000.0*unc47 if np.isfinite(unc47) else np.nan,
        "v47_practical_resolution_ms":1000.0*practical47 if np.isfinite(practical47) else np.nan,
        "master_clock_adjust_s":master_adj,
        "master_clock_correlation":master_corr,
        "master_clock_quality":master_q,
        "front_step_period_s":phase.get("front_step_period_s",np.nan),
        "lateral_step_period_s":phase.get("lateral_step_period_s",np.nan),
        "period_difference_pct":phase.get("period_difference_pct",np.nan),
        "phase_offset_s":phase_off,
        "phase_correction_s":phase.get("phase_correction_from_master_s",0.0),
        "phase_correlation":np.nan,
        "phase_quality":phase_q,
        "phase_pairs":phase.get("phase_pairs",0),
        "phase_mae_s":phase.get("phase_mae_s",np.nan),
        "phase_sequence_error_s":phase.get("phase_sequence_error_s",np.nan),
        "phase_lr_hypothesis":phase.get("phase_lr_hypothesis","No calculable"),
        "phase_supported":1 if phase_supported else 0,
        "lr_hypothesis":"No calculable",
        "direct_pairs":0,"swapped_pairs":0,"sequence_pairs":0,
        "sequence_consistency_ms":np.nan,"second_lateral_homolog":0,
        "single_pair_offset_guard":0,"mae_s":np.nan
    }
    if not A_all or not B_all:
        return [],"No calculable",diag

    A=list(A_all); B=list(B_all)
    if physical_window is not None:
        t0,t1=map(float,physical_window)
        ffps=float(front_snapshot.get("fps",0) or 0)
        lfps=float(lateral_snapshot.get("fps",0) or 0)
        A=[a for a in A if _cycle_overlaps_physical_window(a,ffps,t0,t1,0.0)]
        B=[b for b in B if _cycle_overlaps_physical_window(b,lfps,t0,t1,match_off)]

    def solve(swapped):
        C=[]
        for i,a in enumerate(A):
            for j,b in enumerate(B):
                bs=("R" if b["side"]=="L" else "L") if swapped else b["side"]
                if a["side"]!=bs: continue
                dt=(float(b["ic_s"])-match_off)-float(a["ic_s"])
                da=float(a["next_ic_s"]-a["ic_s"]); db=float(b["next_ic_s"]-b["ic_s"])
                dd=abs(da-db)
                if abs(dt)<=float(tolerance_s) and dd<=0.30:
                    C.append((i,j,float(dt),float(dd)))
        C.sort(key=lambda z:(z[0],z[1]))
        best=[]
        def obj(ps):
            if not ps: return (0,-99,-99,-99)
            e=np.asarray([abs(x[2]) for x in ps],float)
            signed=np.asarray([x[2] for x in ps],float)
            d=np.asarray([x[3] for x in ps],float)
            cons=float(np.std(signed)) if len(ps)>=2 else 0.20
            return (len(ps),-float(np.mean(e)),-cons,-float(np.mean(d)))
        def dfs(k,li,lj,chosen):
            nonlocal best
            if k>=len(C):
                if obj(chosen)>obj(best): best=list(chosen)
                return
            dfs(k+1,li,lj,chosen)
            i,j,dt,dd=C[k]
            if i>li and j>lj:
                chosen.append(C[k]); dfs(k+1,i,j,chosen); chosen.pop()
        if len(C)<=22:
            dfs(0,-1,-1,[])
        else:
            ui=set(); uj=set()
            for x in sorted(C,key=lambda q:abs(q[2])+0.3*q[3]):
                if x[0] not in ui and x[1] not in uj:
                    best.append(x); ui.add(x[0]); uj.add(x[1])
            best.sort()
        pairs=[{"side":A[i]["side"],"front":A[i],"lateral":B[j],"delta_s":dt,
                "duration_delta_s":dd,"lr_swapped":bool(swapped)} for i,j,dt,dd in best]
        mae=float(np.mean([abs(p["delta_s"]) for p in pairs])) if pairs else np.nan
        cons=float(np.std([p["delta_s"] for p in pairs])) if len(pairs)>=2 else np.nan
        return pairs,mae,cons

    pd,md,cd=solve(False); ps,ms,cs=solve(True)
    rankd=(len(pd),-(md if np.isfinite(md) else 9),-(cd if np.isfinite(cd) else 9))
    ranks=(len(ps),-(ms if np.isfinite(ms) else 9),-(cs if np.isfinite(cs) else 9))
    if ranks>rankd:
        pairs,mae,cons,swapped=ps,ms,cs,True
    else:
        pairs,mae,cons,swapped=pd,md,cd,False

    if not pairs:
        quality="No calculable"
    elif len(pairs)>=3 and mae<=0.06 and (not np.isfinite(cons) or cons<=0.05):
        quality="Alta"
    elif len(pairs)>=2 and mae<=0.12 and (not np.isfinite(cons) or cons<=0.10):
        quality="Moderada"
    else:
        quality="Baja"

    lateral_ids={(p["lateral"]["side"],p["lateral"]["ic_frame"]) for p in pairs}

    # V40 · reconstrucción independiente de ciclos desde la secuencia sincronizada de pasos.
    _event_pairs,_triplets,_v40diag=_v40_match_event_sequence(
        front_snapshot,lateral_snapshot,match_off,phase_diag=phase
    )
    _v41diag=_v41_periodic_triplet_rescue(
        front_snapshot,lateral_snapshot,match_off,v40_event_pairs=_event_pairs,phase_diag=phase
    )
    _v44diag=_v44_chain_consecutive_cycles(
        front_snapshot,lateral_snapshot,match_off,phase_diag=phase
    )
    _v45diag=_v45_second_cycle_evaluability(
        front_snapshot,lateral_snapshot,match_off,phase_diag=phase,v44_diag=_v44diag
    )
    diag.update({
        "v45_second_cycle_evaluable":_v45diag.get("second_cycle_evaluable",0),
        "v45_coverage_sufficient":_v45diag.get("coverage_sufficient",0),
        "v45_status":_v45diag.get("status","No evaluable"),
        "v45_reason":_v45diag.get("reason",""),
        "v45_predicted_next_ic_camera_s":_v45diag.get("predicted_next_ic_camera_s",np.nan),
        "v45_predicted_next_ic_physical_s":_v45diag.get("predicted_next_ic_physical_s",np.nan),
        "v45_predicted_next_ic_frame":_v45diag.get("predicted_next_ic_frame",np.nan),
        "v45_visibility_end_frame":_v45diag.get("visibility_end_frame",np.nan),
        "v45_visibility_end_camera_s":_v45diag.get("visibility_end_camera_s",np.nan),
        "v45_visibility_margin_s":_v45diag.get("visibility_margin_s",np.nan),
        "v45_last_accepted_side":_v45diag.get("last_accepted_side",""),
        "v45_last_accepted_frame":_v45diag.get("last_accepted_frame",np.nan),
        "v45_expected_next_side":_v45diag.get("expected_next_side",""),
        "v44_chain_count":_v44diag.get("chain_count",0),
        "v44_chained_cycles":_v44diag.get("cycles_count",0),
        "v44_second_cycle_homolog":_v44diag.get("second_cycle_homolog",0),
        "v44_direct_chains":_v44diag.get("direct_chains",0),
        "v44_swapped_chains":_v44diag.get("swapped_chains",0),
        "v44_endpoint_tolerance_s":_v44diag.get("endpoint_tolerance_s",np.nan),
        "v44_middle_tolerance_s":_v44diag.get("middle_tolerance_s",np.nan),
        "v41_rescue_attempted":_v41diag.get("attempted",0),
        "v41_rescued_events":_v41diag.get("rescued_events",0),
        "v41_triplet_cycles":_v41diag.get("triplet_cycles",0),
        "v41_second_lateral_homolog":_v41diag.get("second_lateral_homolog",0),
        "v41_predicted_step_period_s":_v41diag.get("predicted_step_period_s",np.nan),
        "v41_rescue_tolerance_s":_v41diag.get("rescue_tolerance_s",np.nan),
        "v43_closure_cycles":_v41diag.get("v43_closure_cycles_count",0),
        "v43_second_lateral_homolog":_v41diag.get("v43_second_lateral_homolog",0),
        "v40_event_pairs":_v40diag.get("event_pairs",0),
        "v40_triplet_cycles":_v40diag.get("triplet_cycles",0),
        "v40_second_lateral_homolog":_v40diag.get("second_lateral_homolog",0),
        "v40_event_mae_s":_v40diag.get("event_mae_s",np.nan),
        "v40_event_sequence_error_s":_v40diag.get("event_sequence_error_s",np.nan),
        "v40_lr_hypothesis":_v40diag.get("lr_hypothesis","No calculable"),
        "v40_direct_event_pairs":_v40diag.get("direct_event_pairs",0),
        "v40_swapped_event_pairs":_v40diag.get("swapped_event_pairs",0),
        "v40_temporal_uncertainty_s":_v40diag.get("temporal_uncertainty_s",np.nan),
        "v40_practical_resolution_s":_v40diag.get("practical_resolution_s",np.nan),
        "v40_front_frame_s":_v40diag.get("front_frame_s",np.nan),
        "v40_lateral_frame_s":_v40diag.get("lateral_frame_s",np.nan),
        "v40_front_fps":_v40diag.get("front_fps",np.nan),
        "v40_lateral_fps":_v40diag.get("lateral_fps",np.nan),
        "v40_event_tolerance_s":_v40diag.get("event_tolerance_s",np.nan),
    })
    # Persistir objetos ligeros para la interfaz; los ciclos V40 son tripletes de eventos.
    diag["v40_event_pairs_data"]=_event_pairs
    diag["v40_triplet_data"]=_triplets
    diag["v41_rescue_rows"]=_v41diag.get("rescue_rows",[])
    diag["v41_triplet_data"]=_v41diag.get("triplets",[])
    diag["v42_triplet_audit"]=_v41diag.get("v42_triplet_audit",[])
    diag["v42_candidate_audit"]=_v41diag.get("v42_candidate_audit",[])
    diag["v43_closure_cycle_data"]=_v41diag.get("v43_closure_cycles",[])
    diag["v44_chain_data"]=_v44diag.get("chains",[])
    diag["v44_chain_audit"]=_v44diag.get("audit",[])
    diag["v45_near_prediction_candidates"]=_v45diag.get("rejected_near_prediction",[])

    diag.update({
        "lr_hypothesis":"L↔R invertido" if swapped else "L↔L / R↔R directo",
        "direct_pairs":len(pd),"swapped_pairs":len(ps),
        "direct_mae_s":md,"swapped_mae_s":ms,
        "sequence_pairs":len(pairs),
        "sequence_consistency_ms":float(cons*1000.0) if np.isfinite(cons) else np.nan,
        "second_lateral_homolog":1 if len(lateral_ids)>=2 else 0,
        "mae_s":mae,
        "final_offset_s":match_off
    })
    return pairs,quality,diag



def _v48_resample_normalized_cycle(seg, cycle, columns):
    """Versión 49 · normaliza un ciclo físico IC→IC a 0–100 % y lo remuestrea a 101 puntos."""
    if seg is None or cycle is None:
        return pd.DataFrame()
    try:
        kin = _compute_display_kinematics(seg)
        raw = _normalized_cycle_series(kin, cycle, columns)
    except Exception:
        return pd.DataFrame()
    if raw is None or raw.empty or "cycle_pct" not in raw.columns:
        return pd.DataFrame()
    grid = np.linspace(0.0, 100.0, 101)
    out = pd.DataFrame({"cycle_pct": grid})
    x = pd.to_numeric(raw["cycle_pct"], errors="coerce").to_numpy(float)
    for c in columns:
        if c not in raw.columns:
            continue
        y = pd.to_numeric(raw[c], errors="coerce").to_numpy(float)
        ok = np.isfinite(x) & np.isfinite(y)
        if int(ok.sum()) >= 2:
            xx=x[ok]; yy=y[ok]
            order=np.argsort(xx); xx=xx[order]; yy=yy[order]
            xxu, idx=np.unique(xx, return_index=True); yyu=yy[idx]
            if len(xxu) >= 2:
                out[c]=np.interp(grid,xxu,yyu,left=np.nan,right=np.nan)
    return out



def _v50_norm01(a):
    a=np.asarray(a,dtype=float)
    out=np.full_like(a,np.nan,dtype=float)
    ok=np.isfinite(a)
    if int(ok.sum())<3:
        return out
    lo=float(np.nanpercentile(a[ok],10)); hi=float(np.nanpercentile(a[ok],90))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi-lo<1e-9:
        out[ok]=0.0; return out
    out[ok]=np.clip((a[ok]-lo)/(hi-lo),0.0,1.0)
    return out


def _v50_to_candidate(seg, cycle, fps, side, view_kind):
    """V50 · candidato TO por onset multiseñal distal dentro de un ciclo IC→IC.
    Es una auditoría cinemática 2D; no equivale a fuerza de reacción del suelo.
    """
    empty={"candidate_pct":np.nan,"score":np.nan,"quality":"No calculable","reason":"datos insuficientes","audit":pd.DataFrame()}
    if seg is None or cycle is None or not np.isfinite(float(fps or np.nan)) or float(fps)<=0:
        return empty
    try:
        f0=float(cycle.get("ic_frame",np.nan)); f1=float(cycle.get("next_ic_frame",np.nan))
    except Exception:
        return empty
    if not (np.isfinite(f0) and np.isfinite(f1) and f1>f0): return empty
    fr=pd.to_numeric(seg.get("frame",pd.Series(np.arange(len(seg)))),errors="coerce").to_numpy(float)
    mask=np.isfinite(fr)&(fr>=f0)&(fr<=f1)
    sub=seg.loc[mask].copy()
    if len(sub)<8: return empty
    fr=pd.to_numeric(sub.get("frame",pd.Series(np.arange(len(sub)))),errors="coerce").to_numpy(float)
    pct=(fr-f0)/(f1-f0)*100.0
    side=str(side or "L").upper()[0]
    req=[f"{side}Ankle_x",f"{side}Ankle_y",f"{side}Heel_x",f"{side}Heel_y",f"{side}BigToe_x",f"{side}BigToe_y","LHip_x","LHip_y","RHip_x","RHip_y"]
    if not all(c in sub.columns for c in req):
        empty["reason"]="faltan landmarks distales/pelvis"; return empty
    def a(c): return pd.to_numeric(sub[c],errors="coerce").to_numpy(float)
    foot_x=(a(f"{side}Ankle_x")+a(f"{side}Heel_x")+a(f"{side}BigToe_x"))/3.0
    foot_y=(a(f"{side}Ankle_y")+a(f"{side}Heel_y")+a(f"{side}BigToe_y"))/3.0
    pel_x=(a("LHip_x")+a("RHip_x"))/2.0; pel_y=(a("LHip_y")+a("RHip_y"))/2.0
    # escala corporal para reducir dependencia de distancia a cámara
    if all(c in sub.columns for c in ["LShoulder_y","RShoulder_y"]):
        sh_y=(a("LShoulder_y")+a("RShoulder_y"))/2.0
        scale=np.abs(pel_y-sh_y)
    else:
        scale=np.sqrt((a("LHip_x")-a("RHip_x"))**2+(a("LHip_y")-a("RHip_y"))**2)
    scale=pd.Series(scale).replace([np.inf,-np.inf],np.nan).interpolate(limit_direction="both").to_numpy(float)
    smed=float(np.nanmedian(scale)) if np.isfinite(scale).any() else 1.0
    if not np.isfinite(smed) or smed<3: smed=1.0
    rx=(foot_x-pel_x)/smed; ry=(foot_y-pel_y)/smed
    win=max(3,int(round(0.08*float(fps)))); win += (win%2==0)
    def smooth(v):
        return pd.Series(v).replace([np.inf,-np.inf],np.nan).interpolate(limit_direction="both").rolling(win,center=True,min_periods=1).mean().to_numpy(float)
    rx=smooth(rx); ry=smooth(ry)
    vx=smooth(np.gradient(rx)*float(fps)); vy=smooth(np.gradient(ry)*float(fps))
    speed=smooth(np.sqrt(vx*vx+vy*vy))
    lift=smooth(np.maximum(0.0,-vy))  # coordenada y de imagen: ascenso = y decreciente
    if str(view_kind).lower().startswith("lat"):
        # En lateral, el pie pasa de desplazarse hacia atrás respecto a pelvis a avanzar durante oscilación.
        prog=float(np.nanmedian(np.gradient(smooth(pel_x)))) if np.isfinite(pel_x).any() else 0.0
        direction=1.0 if prog>=0 else -1.0
        forward=smooth(np.maximum(0.0,direction*vx))
        comp=0.45*_v50_norm01(forward)+0.30*_v50_norm01(lift)+0.25*_v50_norm01(speed)
    else:
        # Frontal/posterior: la componente sagital proyectada es débil; se usa como evidencia secundaria.
        comp=0.55*_v50_norm01(speed)+0.35*_v50_norm01(lift)+0.10*_v50_norm01(np.abs(vx))
    comp=smooth(comp)
    region=np.isfinite(pct)&(pct>=35.0)&(pct<=82.0)&np.isfinite(comp)
    if int(region.sum())<4:
        empty["reason"]="ventana TO insuficiente"; return empty
    ridx=np.where(region)[0]
    vals=comp[ridx]
    peak=float(np.nanmax(vals)); base=float(np.nanpercentile(vals,20))
    amp=peak-base
    if not np.isfinite(amp) or amp<0.04:
        empty["reason"]="activación distal insuficiente"; return empty
    thr=base+0.45*amp
    # Primer onset persistente antes del pico: reduce el sesgo hacia el máximo de velocidad de oscilación.
    peak_local=int(ridx[int(np.nanargmax(vals))])
    candidates=[]
    for j in ridx:
        if j>peak_local: break
        if comp[j]>=thr:
            k2=min(len(comp),j+3)
            persist=float(np.nanmean(comp[j:k2]>=thr)) if k2>j else 0.0
            if persist>=0.50: candidates.append(int(j))
    j=int(candidates[0]) if candidates else peak_local
    cand=float(pct[j])
    # Confianza = amplitud del onset + cantidad de datos; frontal queda conservador por geometría.
    completeness=float(np.mean(np.isfinite(comp[region])))
    contrast=float(np.clip(amp/0.45,0.0,1.0))
    score=float(np.clip(0.65*contrast+0.35*completeness,0.0,1.0))
    if str(view_kind).lower().startswith("front"): score*=0.85
    quality="Alta" if score>=0.72 else ("Moderada" if score>=0.48 else "Baja")
    audit=pd.DataFrame({"cycle_pct":pct,"distal_speed_norm":_v50_norm01(speed),"lift_norm":_v50_norm01(lift),"activation":comp})
    audit["candidate_to_pct"]=cand
    return {"candidate_pct":cand,"score":score,"quality":quality,
            "reason":"onset multiseñal distal V50 (35–82% del ciclo)","audit":audit}


def _build_v50_to_consensus(seg_front, seg_lateral, fc, lc, fps_front, fps_lateral, front_side, lateral_side, original_front_to, original_lateral_to):
    """V50 · audita TO en ambos planos sin modificar IC, reloj físico ni ciclos V47."""
    vf=_v50_to_candidate(seg_front,fc,fps_front,front_side,"front")
    vl=_v50_to_candidate(seg_lateral,lc,fps_lateral,lateral_side,"lateral")
    fp=float(vf.get("candidate_pct",np.nan)); lp=float(vl.get("candidate_pct",np.nan))
    fs=float(vf.get("score",np.nan)); ls=float(vl.get("score",np.nan))
    delta=abs(fp-lp) if np.isfinite(fp) and np.isfinite(lp) else np.nan
    lat_vs_old=abs(lp-float(original_lateral_to)) if np.isfinite(lp) and np.isfinite(float(original_lateral_to)) else np.nan
    front_vs_old=abs(fp-float(original_front_to)) if np.isfinite(fp) and np.isfinite(float(original_front_to)) else np.nan
    confirmed=np.nan; status="No calculable"; quality="No calculable"
    if np.isfinite(delta):
        if delta<=8.0 and min(fs if np.isfinite(fs) else 0,ls if np.isfinite(ls) else 0)>=0.48:
            confirmed=float(np.average([fp,lp],weights=[max(fs,0.05),max(ls,0.05)]))
            status="Confirmado biplanar"
            quality="Alta" if delta<=5.0 and min(fs,ls)>=0.65 else "Moderada"
        elif delta<=12.0:
            status="Concordancia limitada"
            quality="Moderada"
        elif np.isfinite(lat_vs_old) and lat_vs_old<=8.0 and np.isfinite(ls) and ls>=0.48:
            status="Respaldado lateralmente; no confirmado biplanar"
            quality="Lateral"
        else:
            status="Discrepante; no confirmado"
            quality="No confirmado"
    audits=[]
    for view,obj in [("Frontal/posterior",vf),("Lateral",vl)]:
        ad=obj.get("audit")
        if isinstance(ad,pd.DataFrame) and not ad.empty:
            q=ad.copy(); q.insert(0,"view",view); audits.append(q)
    audit=pd.concat(audits,ignore_index=True) if audits else pd.DataFrame()
    summary={
        "v50_front_to_candidate_pct":fp,"v50_lateral_to_candidate_pct":lp,
        "v50_front_to_score":fs,"v50_lateral_to_score":ls,
        "v50_front_to_quality":str(vf.get("quality","No calculable")),
        "v50_lateral_to_quality":str(vl.get("quality","No calculable")),
        "v50_candidate_delta_pct":float(delta) if np.isfinite(delta) else np.nan,
        "v50_front_vs_v49_delta_pct":float(front_vs_old) if np.isfinite(front_vs_old) else np.nan,
        "v50_lateral_vs_v49_delta_pct":float(lat_vs_old) if np.isfinite(lat_vs_old) else np.nan,
        "v50_to_status":status,"v50_to_quality":quality,
        "v50_biplanar_to_confirmed_pct":float(confirmed) if np.isfinite(confirmed) else np.nan,
        "v50_method":"Onset distal multiseñal independiente por vista; TO solo se confirma si ambas vistas convergen. No modifica reloj V47 ni IC→IC."
    }
    return {"summary":summary,"audit":audit}



def _v51_interp_audit_value(audit, view_name, pct, col="activation"):
    """Interpola una señal auditada V50 en un porcentaje concreto del ciclo."""
    if not isinstance(audit,pd.DataFrame) or audit.empty or not np.isfinite(float(pct)):
        return np.nan
    q=audit[audit.get("view",pd.Series(index=audit.index,dtype=object)).astype(str)==str(view_name)].copy()
    if q.empty or "cycle_pct" not in q.columns or col not in q.columns:
        return np.nan
    x=pd.to_numeric(q["cycle_pct"],errors="coerce").to_numpy(float)
    y=pd.to_numeric(q[col],errors="coerce").to_numpy(float)
    ok=np.isfinite(x)&np.isfinite(y)
    if int(ok.sum())<2: return np.nan
    x=x[ok]; y=y[ok]; order=np.argsort(x); x=x[order]; y=y[order]
    return float(np.interp(float(pct),x,y,left=np.nan,right=np.nan))


def _build_v51_to_hierarchy(v50, cycle_lateral_s=np.nan, fps_lateral=np.nan):
    """V51 · detector TO jerárquico dependiente de la perspectiva.

    La vista lateral es primaria para TO por contener mejor la progresión sagital del pie.
    La vista frontal/posterior se utiliza como evidencia secundaria de compatibilidad y
    nunca desplaza el reloj físico V47 ni modifica IC→IC. Los umbrales son internos y
    experimentales; no equivalen a validación con plataforma de fuerzas.
    """
    if not isinstance(v50,dict):
        return {"summary":{"v51_to_status":"No calculable","v51_to_quality":"No calculable"},"audit":pd.DataFrame()}
    vs=v50.get("summary") or {}; audit=v50.get("audit")
    fp=float(vs.get("v50_front_to_candidate_pct",np.nan)); lp=float(vs.get("v50_lateral_to_candidate_pct",np.nan))
    fs=float(vs.get("v50_front_to_score",np.nan)); ls=float(vs.get("v50_lateral_to_score",np.nan))
    delta=abs(fp-lp) if np.isfinite(fp) and np.isfinite(lp) else np.nan

    # Evidencia frontal exactamente en el instante propuesto por la vista lateral.
    f_act=_v51_interp_audit_value(audit,"Frontal/posterior",lp,"activation") if np.isfinite(lp) else np.nan
    f_speed=_v51_interp_audit_value(audit,"Frontal/posterior",lp,"distal_speed_norm") if np.isfinite(lp) else np.nan
    f_lift=_v51_interp_audit_value(audit,"Frontal/posterior",lp,"lift_norm") if np.isfinite(lp) else np.nan
    f_peak=np.nan
    if isinstance(audit,pd.DataFrame) and not audit.empty and "view" in audit.columns and "activation" in audit.columns:
        q=audit[audit["view"].astype(str)=="Frontal/posterior"]
        if not q.empty:
            vals=pd.to_numeric(q["activation"],errors="coerce").to_numpy(float)
            if np.isfinite(vals).any(): f_peak=float(np.nanmax(vals))
    f_support=float(f_act/f_peak) if np.isfinite(f_act) and np.isfinite(f_peak) and f_peak>1e-9 else np.nan

    # Resolución temporal de la vista primaria expresada en % del ciclo.
    frame_pct=np.nan
    if np.isfinite(float(cycle_lateral_s)) and float(cycle_lateral_s)>0 and np.isfinite(float(fps_lateral)) and float(fps_lateral)>0:
        frame_pct=100.0/(float(cycle_lateral_s)*float(fps_lateral))
    direct_tol=max(8.0,2.0*frame_pct) if np.isfinite(frame_pct) else 8.0

    selected=np.nan; status="No calculable"; quality="No calculable"; front_role="No evaluable"; confirmed=0.0
    lateral_ok=np.isfinite(lp) and np.isfinite(ls) and ls>=0.48 and 35.0<=lp<=82.0
    if lateral_ok:
        selected=float(lp)  # V51: la vista lateral gobierna el instante TO.
        if np.isfinite(delta) and np.isfinite(fs) and fs>=0.48 and delta<=direct_tol:
            status="Confirmado biplanar directo"
            quality="Alta" if ls>=0.65 and fs>=0.65 and delta<=max(5.0,frame_pct if np.isfinite(frame_pct) else 5.0) else "Moderada"
            front_role="Candidato frontal concordante"
            confirmed=1.0
        elif np.isfinite(f_support) and f_support>=0.45:
            status="Confirmado jerárquico"
            quality="Alta" if ls>=0.72 and f_support>=0.60 else "Moderada"
            front_role="Evidencia frontal compatible en el TO lateral"
            confirmed=1.0
        elif np.isfinite(f_support) and f_support>=0.25:
            status="TO lateral probable; frontal no concluyente"
            quality="Moderada" if ls>=0.65 else "Limitada"
            front_role="Evidencia frontal débil/no concluyente"
            confirmed=0.0
        else:
            status="TO lateral no confirmado por frontal"
            quality="Limitada"
            front_role="Sin respaldo frontal suficiente"
            confirmed=0.0
    elif np.isfinite(lp):
        status="Candidato lateral de baja calidad"
        quality="Baja"
        front_role="No usado para confirmar"

    # Tiempo desde IC y ventana práctica ±1 frame de la cámara lateral.
    selected_s=float(selected/100.0*float(cycle_lateral_s)) if np.isfinite(selected) and np.isfinite(float(cycle_lateral_s)) else np.nan
    unc_ms=1000.0/float(fps_lateral) if np.isfinite(float(fps_lateral)) and float(fps_lateral)>0 else np.nan
    lo=max(0.0,selected-(frame_pct if np.isfinite(frame_pct) else 0.0)) if np.isfinite(selected) else np.nan
    hi=min(100.0,selected+(frame_pct if np.isfinite(frame_pct) else 0.0)) if np.isfinite(selected) else np.nan

    summary={
        "v51_primary_view":"Lateral/sagital",
        "v51_secondary_view":"Frontal/posterior",
        "v51_lateral_primary_to_pct":float(lp) if np.isfinite(lp) else np.nan,
        "v51_front_candidate_pct":float(fp) if np.isfinite(fp) else np.nan,
        "v51_candidate_delta_pct":float(delta) if np.isfinite(delta) else np.nan,
        "v51_front_support_at_lateral_to":float(f_support) if np.isfinite(f_support) else np.nan,
        "v51_front_activation_at_lateral_to":float(f_act) if np.isfinite(f_act) else np.nan,
        "v51_front_speed_at_lateral_to":float(f_speed) if np.isfinite(f_speed) else np.nan,
        "v51_front_lift_at_lateral_to":float(f_lift) if np.isfinite(f_lift) else np.nan,
        "v51_direct_tolerance_pct":float(direct_tol),
        "v51_lateral_frame_resolution_pct":float(frame_pct) if np.isfinite(frame_pct) else np.nan,
        "v51_selected_to_pct":float(selected) if np.isfinite(selected) else np.nan,
        "v51_selected_to_from_ic_s":float(selected_s) if np.isfinite(selected_s) else np.nan,
        "v51_selected_to_uncertainty_ms":float(unc_ms) if np.isfinite(unc_ms) else np.nan,
        "v51_selected_to_window_low_pct":float(lo) if np.isfinite(lo) else np.nan,
        "v51_selected_to_window_high_pct":float(hi) if np.isfinite(hi) else np.nan,
        "v51_to_confirmed":float(confirmed),
        "v51_to_status":status,
        "v51_to_quality":quality,
        "v51_front_role":front_role,
        "v51_method":"Detector jerárquico: TO lateral como referencia primaria; frontal/posterior como evidencia secundaria de compatibilidad. No promedia candidatos discordantes, no cambia V47 ni IC→IC; umbrales internos experimentales 2D."
    }
    rows=[{
        "primary_view":"Lateral/sagital","secondary_view":"Frontal/posterior",
        "lateral_to_pct":lp,"front_candidate_pct":fp,"delta_pct":delta,
        "front_support_ratio_at_lateral_to":f_support,"front_activation_at_lateral_to":f_act,
        "front_speed_norm_at_lateral_to":f_speed,"front_lift_norm_at_lateral_to":f_lift,
        "direct_tolerance_pct":direct_tol,"selected_to_pct":selected,
        "selected_to_from_ic_s":selected_s,"uncertainty_ms":unc_ms,
        "status":status,"quality":quality,"front_role":front_role,"confirmed":confirmed
    }]
    return {"summary":summary,"audit":pd.DataFrame(rows)}

def _build_v52_clinical_cycle(curves, summary):
    """V52 · convierte el ciclo homólogo ya validado en una línea temporal clínica IC→TO→IC.

    Usa exclusivamente el TO seleccionado por V51 y NO modifica reloj V47, IC, emparejamiento
    de ciclo ni detectores previos. Resume apoyo/oscilación y describe cada curva por fase sin
    aplicar rangos normativos ni inferir cinética.
    """
    if not isinstance(curves,pd.DataFrame) or curves.empty or not isinstance(summary,dict):
        return {"summary":{"v52_status":"No calculable"},"phase_metrics":pd.DataFrame(),"events":pd.DataFrame(),"curves":curves}
    to_pct=float(summary.get("v51_selected_to_pct",np.nan))
    to_confirmed=float(summary.get("v51_to_confirmed",0.0) or 0.0)
    to_status=str(summary.get("v51_to_status","No calculable"))
    to_quality=str(summary.get("v51_to_quality","No calculable"))
    cycle_s=float(summary.get("lateral_cycle_s",np.nan))
    if not np.isfinite(to_pct) or not (0.0 < to_pct < 100.0):
        return {"summary":{"v52_status":"No calculable","v52_reason":"TO V51 no disponible"},"phase_metrics":pd.DataFrame(),"events":pd.DataFrame(),"curves":curves.copy()}

    stance_pct=float(to_pct); swing_pct=float(100.0-to_pct)
    stance_s=float(cycle_s*stance_pct/100.0) if np.isfinite(cycle_s) else np.nan
    swing_s=float(cycle_s*swing_pct/100.0) if np.isfinite(cycle_s) else np.nan
    clinical_status="Ciclo clínico completo" if to_confirmed>=0.5 else "Ciclo con TO probable"
    quality=to_quality if to_quality not in ("", "No calculable") else str(summary.get("integration_confidence","No calculable"))

    c=curves.copy()
    pct=pd.to_numeric(c.get("cycle_pct"),errors="coerce")
    c["Fase V52"]=np.where(pct < stance_pct,"Apoyo","Oscilación")
    c.loc[np.isclose(pct,0.0,atol=0.51),"Evento V52"]="IC (Contacto Inicial)"
    c.loc[np.isclose(pct,100.0,atol=0.51),"Evento V52"]="Siguiente IC"
    if "Evento V52" not in c.columns: c["Evento V52"]=""
    idx_to=(pct-stance_pct).abs().idxmin() if pct.notna().any() else None
    if idx_to is not None: c.loc[idx_to,"Evento V52"]="TO (Despegue del Pie)"
    c["TO seleccionado V51 (%)"]=stance_pct

    numeric_cols=[]
    for col in c.columns:
        if str(col).startswith("Frontal ·") or str(col).startswith("Lateral ·"):
            numeric_cols.append(col)
    rows=[]
    x=pct.to_numpy(float)
    for col in numeric_cols:
        y=pd.to_numeric(c[col],errors="coerce").to_numpy(float)
        ok=np.isfinite(x)&np.isfinite(y)
        if int(ok.sum())<2: continue
        ic=float(np.interp(0.0,x[ok],y[ok])); tov=float(np.interp(stance_pct,x[ok],y[ok])); nic=float(np.interp(100.0,x[ok],y[ok]))
        for phase,mask in [("Apoyo",ok&(x<=stance_pct)),("Oscilación",ok&(x>=stance_pct))]:
            vals=y[mask]
            if vals.size<2: continue
            rows.append({
                "variable":col,"fase":phase,"media":float(np.nanmean(vals)),"min":float(np.nanmin(vals)),
                "max":float(np.nanmax(vals)),"ROM":float(np.nanmax(vals)-np.nanmin(vals)),
                "valor_IC":ic,"valor_TO":tov,"valor_siguiente_IC":nic,
            })
    phase_metrics=pd.DataFrame(rows)
    events=pd.DataFrame([
        {"evento":"IC (Initial Contact / Contacto Inicial)","cycle_pct":0.0,"time_from_ic_s":0.0,"source":"Ciclo homólogo V47"},
        {"evento":"TO (Toe Off / Despegue del Pie)","cycle_pct":stance_pct,"time_from_ic_s":stance_s,"source":"Detector jerárquico V51"},
        {"evento":"Siguiente IC","cycle_pct":100.0,"time_from_ic_s":cycle_s,"source":"Ciclo homólogo V47"},
    ])
    s={
        "v52_status":clinical_status,"v52_quality":quality,"v52_to_confirmed":to_confirmed,
        "v52_to_status_source":to_status,"v52_to_pct":stance_pct,"v52_stance_pct":stance_pct,"v52_swing_pct":swing_pct,
        "v52_cycle_duration_s":cycle_s,"v52_stance_duration_s":stance_s,"v52_swing_duration_s":swing_s,
        "v52_variables_total":len(numeric_cols),"v52_phase_metric_rows":len(phase_metrics),
        "v52_method":"Ciclo clínico biplanar 2D IC→apoyo→TO→oscilación→IC. Usa reloj/ciclo V47 y TO jerárquico V51 sin recalcularlos; sin inferencia cinética ni equivalencia 3D."
    }
    return {"summary":s,"phase_metrics":phase_metrics,"events":events,"curves":c}


def _v52_timeline_figure(to_pct):
    if not np.isfinite(float(to_pct)): return None
    fig,ax=plt.subplots(figsize=(10,1.8))
    ax.plot([0,100],[0,0],linewidth=8,solid_capstyle="butt")
    ax.axvline(0,linewidth=1.5); ax.axvline(float(to_pct),linewidth=1.5); ax.axvline(100,linewidth=1.5)
    ax.text(0,0.18,"IC",ha="left",va="bottom"); ax.text(float(to_pct),0.18,"TO",ha="center",va="bottom"); ax.text(100,0.18,"IC",ha="right",va="bottom")
    ax.text(float(to_pct)/2,-0.18,"Apoyo",ha="center",va="top"); ax.text((float(to_pct)+100)/2,-0.18,"Oscilación",ha="center",va="top")
    ax.set_xlim(0,100); ax.set_ylim(-0.45,0.45); ax.set_yticks([]); ax.set_xlabel("Ciclo de marcha (%)")
    ax.spines[["left","right","top"]].set_visible(False); fig.tight_layout()
    return fig


def _build_v48_biplanar_profile(pairs, seg_front, seg_lateral, fps_front, fps_lateral, pair_quality):
    """
    Versión 49 · integra un mismo ciclo físico visto en frontal/posterior y lateral.
    No triangula 3D: conserva cada variable en su plano 2D y comparte únicamente el eje de fase 0–100 %.
    """
    if not pairs or seg_front is None or seg_lateral is None:
        return None
    valid=[]
    for i,p in enumerate(pairs):
        try:
            dt=abs(float(p.get("delta_s",np.nan)))
            dd=abs(float(p.get("duration_delta_s",np.nan)))
            cost=(dt if np.isfinite(dt) else 9.0)+0.50*(dd if np.isfinite(dd) else 9.0)
            valid.append((cost,i,p))
        except Exception:
            continue
    if not valid:
        return None
    _, pair_index, p=min(valid,key=lambda z:z[0])
    fc=dict(p.get("front") or {}); lc=dict(p.get("lateral") or {})
    if not fc or not lc:
        return None
    front_side=str(fc.get("side",p.get("side","?")))
    lateral_side=str(lc.get("side","?"))
    front_name="Izquierda" if front_side=="L" else "Derecha"
    lateral_name="Izquierda" if lateral_side=="L" else "Derecha"

    front_cols=[
        "Oblicuidad pélvica (°)",
        "Inclinación tronco (°)",
        f"Desviación frontal rodilla {front_name} (°)",
        f"Orientación pie {front_name} (°)",
        f"Retropié {front_name} (°)",
    ]
    lateral_cols=[
        f"Flexión cadera {lateral_name} (°)",
        f"Flexión rodilla {lateral_name} (°)",
        f"Ángulo tobillo {lateral_name} (°)",
    ]
    nf=_v48_resample_normalized_cycle(seg_front,fc,front_cols)
    nl=_v48_resample_normalized_cycle(seg_lateral,lc,lateral_cols)
    if nf.empty and nl.empty:
        return None
    combined=pd.DataFrame({"cycle_pct":np.linspace(0.0,100.0,101)})
    for c in front_cols:
        if c in nf.columns:
            combined[f"Frontal · {c}"]=nf[c].to_numpy(float)
    for c in lateral_cols:
        if c in nl.columns:
            combined[f"Lateral · {c}"]=nl[c].to_numpy(float)

    fcy=(float(fc.get("next_ic_frame",np.nan))-float(fc.get("ic_frame",np.nan)))/float(fps_front) if fps_front else np.nan
    lcy=(float(lc.get("next_ic_frame",np.nan))-float(lc.get("ic_frame",np.nan)))/float(fps_lateral) if fps_lateral else np.nan
    fstance=float(fc.get("stance_pct",np.nan)); lstance=float(lc.get("stance_pct",np.nan))
    # V49 · QC explícito de TO entre planos. IC fija el ciclo físico; TO es un
    # evento cinemático estimado de forma independiente en cada vista y no debe
    # asumirse como homólogo si ambos detectores discrepan.
    to_delta_pct = abs(fstance-lstance) if np.isfinite(fstance) and np.isfinite(lstance) else np.nan
    if np.isfinite(to_delta_pct):
        if to_delta_pct <= 8.0:
            to_status="Concordante"
            to_quality="Alta"
            to_confirmed_pct=float((fstance+lstance)/2.0)
        elif to_delta_pct <= 15.0:
            to_status="Concordancia limitada"
            to_quality="Moderada"
            to_confirmed_pct=np.nan
        else:
            to_status="Discrepante"
            to_quality="No confirmado"
            to_confirmed_pct=np.nan
    else:
        to_status="No calculable"
        to_quality="No calculable"
        to_confirmed_pct=np.nan
    delta_ms=1000.0*float(p.get("delta_s",np.nan))
    dur_delta_ms=1000.0*float(p.get("duration_delta_s",np.nan))
    quality=str(pair_quality or "No calculable")
    if quality=="Alta": confidence="Alta"
    elif quality=="Moderada": confidence="Moderada"
    else: confidence="Limitada"
    summary={
        "selected_pair_index":int(pair_index+1),
        "front_side":front_side,
        "lateral_side":lateral_side,
        "lr_swapped":bool(p.get("lr_swapped",False)),
        "pair_quality":quality,
        "integration_confidence":confidence,
        "front_ic_frame":int(fc.get("ic_frame",-1)),
        "front_to_frame":int(fc.get("to_frame",-1)),
        "front_next_ic_frame":int(fc.get("next_ic_frame",-1)),
        "lateral_ic_frame":int(lc.get("ic_frame",-1)),
        "lateral_to_frame":int(lc.get("to_frame",-1)),
        "lateral_next_ic_frame":int(lc.get("next_ic_frame",-1)),
        "front_cycle_s":fcy,
        "lateral_cycle_s":lcy,
        "ic_alignment_delta_ms":delta_ms,
        "cycle_duration_delta_ms":dur_delta_ms,
        "front_to_pct":fstance,
        "lateral_to_pct":lstance,
        "to_delta_pct":float(to_delta_pct) if np.isfinite(to_delta_pct) else np.nan,
        "to_concordance_status":to_status,
        "to_concordance_quality":to_quality,
        "biplanar_to_confirmed_pct":float(to_confirmed_pct) if np.isfinite(to_confirmed_pct) else np.nan,
        "variables_front":int(len([c for c in combined.columns if c.startswith("Frontal ·")])),
        "variables_lateral":int(len([c for c in combined.columns if c.startswith("Lateral ·")])),
        "method":"Ciclo físico homólogo normalizado 0–100%; integración multiplanar 2D, no reconstrucción 3D.",
    }
    v50=_build_v50_to_consensus(seg_front,seg_lateral,fc,lc,fps_front,fps_lateral,front_side,lateral_side,fstance,lstance)
    if isinstance(v50,dict):
        summary.update(v50.get("summary") or {})
    v51=_build_v51_to_hierarchy(v50,lcy,fps_lateral)
    if isinstance(v51,dict):
        summary.update(v51.get("summary") or {})
    v52=_build_v52_clinical_cycle(combined,summary)
    if isinstance(v52,dict):
        summary.update(v52.get("summary") or {})
        combined=v52.get("curves") if isinstance(v52.get("curves"),pd.DataFrame) else combined
    return {"summary":summary,"curves":combined,"pair":p,
            "v50_to_audit":(v50.get("audit") if isinstance(v50,dict) else pd.DataFrame()),
            "v51_to_audit":(v51.get("audit") if isinstance(v51,dict) else pd.DataFrame()),
            "v52_phase_metrics":(v52.get("phase_metrics") if isinstance(v52,dict) else pd.DataFrame()),
            "v52_events":(v52.get("events") if isinstance(v52,dict) else pd.DataFrame())}


def _v48_profile_figure(curves, columns, title, front_to_pct=np.nan, lateral_to_pct=np.nan):
    """Figura V49 con eje de fase común y eventos IC/TO visibles."""
    if curves is None or curves.empty or not columns:
        return None
    fig,ax=plt.subplots(figsize=(8.0,3.2))
    x=pd.to_numeric(curves["cycle_pct"],errors="coerce").to_numpy(float)
    plotted=False
    for c in columns:
        if c not in curves.columns:
            continue
        y=pd.to_numeric(curves[c],errors="coerce").to_numpy(float)
        ok=np.isfinite(x)&np.isfinite(y)
        if int(ok.sum())>=2:
            ax.plot(x[ok],y[ok],linewidth=1.6,label=c)
            plotted=True
    ax.axvline(0.0,linestyle="--",linewidth=1.0,label="IC (Contacto Inicial)")
    ax.axvline(100.0,linestyle="--",linewidth=1.0)
    if np.isfinite(front_to_pct):
        ax.axvline(float(front_to_pct),linestyle=":",linewidth=1.3,label=f"TO frontal {front_to_pct:.1f}%")
    if np.isfinite(lateral_to_pct):
        ax.axvline(float(lateral_to_pct),linestyle="-.",linewidth=1.3,label=f"TO lateral {lateral_to_pct:.1f}%")
    ax.set_xlim(0,100); ax.set_xlabel("Ciclo de marcha normalizado (%)"); ax.set_ylabel("Ángulo / magnitud 2D (°)")
    ax.set_title(title); ax.grid(True,alpha=0.20)

    # V64 · leyenda realmente EXTERNA al eje de datos (Pestaña 9).
    # La leyenda se dibuja a nivel de FIGURA, en una banda inferior propia.
    # Así nunca comparte espacio con el eje X, su etiqueta, las curvas ni IC/TO.
    handles, labels = ax.get_legend_handles_labels()
    if plotted and handles:
        n_items = len(handles)
        ncol = 2 if n_items >= 2 else 1
        nrows = int(np.ceil(n_items / ncol))

        # Altura adicional dedicada exclusivamente a la leyenda.
        legend_band = 0.48 + 0.28 * nrows
        fig.set_size_inches(8.0, 3.2 + legend_band, forward=True)

        # Reservar primero una banda inferior amplia: el eje X termina claramente
        # por encima de la leyenda. No usar tight_layout después de fig.legend,
        # porque podría volver a comprimir ambos espacios.
        bottom_frac = min(0.46, 0.20 + 0.055 * nrows)
        fig.subplots_adjust(left=0.11, right=0.98, top=0.88, bottom=bottom_frac)

        fig.legend(
            handles, labels,
            loc="lower center",
            bbox_to_anchor=(0.5, 0.025),
            bbox_transform=fig.transFigure,
            fontsize=6.5,
            ncol=ncol,
            frameon=False,
            borderaxespad=0.0,
            columnspacing=1.4,
            handlelength=2.0,
        )
    else:
        fig.tight_layout()
    return fig

def prefix_metrics(metrics, prefix, label_prefix):
    out = []
    for m in metrics:
        x = dict(m)
        x["key"] = f"{prefix}_{m['key']}"
        x["label"] = f"{label_prefix} · {m['label']}"
        out.append(x)
    return out


def camera_pose_quality(df):
    if df is None or df.empty:
        return {"tracking": np.nan, "good_frames": np.nan, "foot_visibility": np.nan, "lower_visibility": np.nan}
    score_cols = [f"{n}_score" for n in LOWER_BODY if f"{n}_score" in df.columns]
    tracking = float(df[score_cols].mean(axis=1).mean()) if score_cols else np.nan
    good = float((df[score_cols].min(axis=1) >= 0.5).mean() * 100) if score_cols else np.nan
    foot = visibility_pct(df, FOOT_POINTS, 0.5)
    lower = visibility_pct(df, LOWER_BODY, 0.5)
    return {"tracking": tracking, "good_frames": good, "foot_visibility": foot, "lower_visibility": lower}


def readiness_3d(mode, q1, q2, sync_quality, calibration_name):
    reasons = []
    if not mode.startswith("2 cámaras"):
        reasons.append("Se requieren dos cámaras.")
    if not calibration_name:
        reasons.append("Falta un perfil de calibración 2 cámaras.")
    if sync_quality not in ("Alta", "Moderada"):
        reasons.append("La sincronización automática es insuficiente o no calculable.")
    for idx, q in enumerate((q1, q2), start=1):
        lv = q.get("lower_visibility", np.nan) if q else np.nan
        if not np.isfinite(lv) or lv < 70:
            reasons.append(f"Visibilidad de tren inferior insuficiente en cámara {idx} (<70%).")
    return len(reasons) == 0, reasons



def _edges(mask):
    """Devuelve inicios y finales de periodos True. Índices relativos al segmento."""
    m=np.asarray(mask,dtype=bool)
    if len(m)==0:
        return np.array([],dtype=int), np.array([],dtype=int)
    d=np.diff(m.astype(int), prepend=int(m[0]))
    starts=np.where(d==1)[0].tolist()
    ends=np.where(d==-1)[0].tolist()
    if m[0]: starts=[0]+starts
    if m[-1]: ends=ends+[len(m)]
    return np.asarray(starts,dtype=int), np.asarray(ends,dtype=int)


def _nearest_valid_indices(starts, ends, n):
    out=[]
    for ic in starts:
        offs=ends[ends>ic]
        if len(offs):
            out.append((int(ic), int(min(offs[0], n-1))))
    return out


def _mean_at_indices(arr, idxs):
    a=np.asarray(arr,float)
    vals=[a[int(i)] for i in idxs if 0<=int(i)<len(a) and np.isfinite(a[int(i)])]
    return float(np.nanmean(vals)) if vals else np.nan


def _window_stat(arr, starts, cycles, lo_pct, hi_pct, fn="max"):
    """Estadística en ventana porcentual del ciclo, usando IC consecutivos como ciclo."""
    a=np.asarray(arr,float); vals=[]
    for j in range(len(starts)-1):
        i0,i1=int(starts[j]),int(starts[j+1])
        if i1-i0<5: continue
        lo=i0+int(round((i1-i0)*lo_pct/100.0)); hi=i0+int(round((i1-i0)*hi_pct/100.0))
        lo=max(i0,min(lo,len(a)-1)); hi=max(lo+1,min(hi,len(a)))
        x=a[lo:hi]; x=x[np.isfinite(x)]
        if not len(x): continue
        vals.append(float(np.nanmax(x) if fn=="max" else np.nanmin(x) if fn=="min" else np.nanmean(x)))
    return float(np.nanmean(vals)) if vals else np.nan



def compute_gait_phase_metrics(seg, fps, is_frontal, expected_stride_s=np.nan,
                               support_summary_l=None, support_summary_r=None,
                               temporal_coherence_ok=True):
    """
    v0.9.2: fases del ciclo a partir de estados de apoyo estabilizados.
    IC/TO siguen siendo eventos cinemáticos 2D estimados.
    """
    lmask = np.asarray(seg["L_support_2d"], bool) if "L_support_2d" in seg else _support_mask_2d(seg, "L", fps, expected_stride_s)[0]
    rmask = np.asarray(seg["R_support_2d"], bool) if "R_support_2d" in seg else _support_mask_2d(seg, "R", fps, expected_stride_s)[0]

    L = support_summary_l or _support_cycle_summary(lmask, fps, expected_stride_s)
    R = support_summary_r or _support_cycle_summary(rmask, fps, expected_stride_s)
    reliable = L["quality"] in ("Alta", "Moderada") and R["quality"] in ("Alta", "Moderada")

    if reliable:
        # v0.9.2: el doble apoyo se calcula SOLO en el dominio cubierto
        # simultáneamente por ciclos válidos de ambos lados y por marcha rectilínea.
        cov_l = np.asarray(L.get("coverage_mask", np.ones(len(lmask), dtype=bool)), bool)
        cov_r = np.asarray(R.get("coverage_mask", np.ones(len(rmask), dtype=bool)), bool)
        valid_domain = cov_l & cov_r
        if "straight_walking_valid" in seg:
            valid_domain &= np.asarray(seg["straight_walking_valid"], bool)

        both = lmask & rmask
        none = (~lmask) & (~rmask)

        if valid_domain.any():
            double_pct_raw = float(np.mean(both[valid_domain]) * 100.0)
            double_s_raw = float(np.sum(both[valid_domain]) / float(fps))
            flight_pct = float(np.mean(none[valid_domain]) * 100.0)
        else:
            double_pct_raw = double_s_raw = flight_pct = np.nan

        swing_l = float(np.nanmean(L["swing"])) if len(L["swing"]) else np.nan
        swing_r = float(np.nanmean(R["swing"])) if len(R["swing"]) else np.nan
        swing_asym = abs(swing_l - swing_r) / ((swing_l + swing_r) / 2.0) * 100.0 if np.isfinite(swing_l) and np.isfinite(swing_r) and (swing_l + swing_r) > 0 else np.nan
        stance_pct_l = float(np.nanmean(L["stance_pct"])) if len(L["stance_pct"]) else np.nan
        stance_pct_r = float(np.nanmean(R["stance_pct"])) if len(R["stance_pct"]) else np.nan
        stance_pct = float(np.nanmean([stance_pct_l, stance_pct_r])) if np.isfinite(stance_pct_l) and np.isfinite(stance_pct_r) else np.nan
        swing_pct = 100.0 - stance_pct if np.isfinite(stance_pct) else np.nan

        # Control físico interno: en marcha ordinaria el detector no debería
        # producir una fase aérea apreciable. Si ocurre, no se publica doble apoyo.
        expected_double = max(0.0, stance_pct_l + stance_pct_r - 100.0) if np.isfinite(stance_pct_l) and np.isfinite(stance_pct_r) else np.nan
        ds_discrepancy = abs(double_pct_raw - expected_double) if np.isfinite(double_pct_raw) and np.isfinite(expected_double) else np.nan
        ds_ok = (
            np.isfinite(double_pct_raw)
            and np.isfinite(flight_pct)
            and flight_pct <= 5.0
            and (not np.isfinite(ds_discrepancy) or ds_discrepancy <= 5.0)
            and bool(temporal_coherence_ok)
        )
        if ds_ok:
            double_pct = double_pct_raw
            double_s = double_s_raw
            phase_quality = "Experimental · ciclos rectilíneos validados"
        else:
            double_pct = double_s = np.nan
            phase_quality = "Parcialmente fiable · doble apoyo suprimido por coherencia temporal/física"
    else:
        double_pct = double_s = swing_l = swing_r = swing_asym = stance_pct = swing_pct = np.nan
        flight_pct = expected_double = ds_discrepancy = np.nan
        phase_quality = "No fiable"

    out = [
        {"key":"temporal_segmentation_score","label":"Consistencia de segmentación apoyo/oscilación","value":float(min(L["score"], R["score"])),"unit":"/100","quality":"Alta" if reliable and min(L["score"],R["score"])>=75 else ("Moderada" if reliable else "No fiable"),"notes":f"Control ciclo a ciclo. Izq: {L['reason']}. Der: {R['reason']}. No es un índice clínico."},
        {"key":"stance_pct_2d","label":"Fase de apoyo estimada","value":stance_pct,"unit":"% ciclo","quality":phase_quality,"notes":"IC→TO estimados mediante estados temporales estabilizados; referencia ~60% solo como contexto."},
        {"key":"swing_pct_2d","label":"Fase de oscilación estimada","value":swing_pct,"unit":"% ciclo","quality":phase_quality,"notes":"TO→siguiente IC; se anula si falla la consistencia temporal."},
        {"key":"double_support_pct_2d","label":"Doble apoyo estimado","value":double_pct,"unit":"% dominio válido","quality":phase_quality,"notes":"v0.9.2: intersección directa de apoyos dentro del dominio rectilíneo común. Se suprime si hay fase aérea >5%, discordancia >5 puntos porcentuales respecto a la ocupación de apoyo o fallo de coherencia temporal global."},
        {"key":"double_support_expected_from_stance_pct","label":"Doble apoyo esperado por ocupación de apoyo","value":expected_double,"unit":"% ciclo","quality":"Control interno","notes":"Máx(0, apoyo_I% + apoyo_D% − 100). Se usa como comprobación matemática, no como medida clínica independiente."},
        {"key":"unsupported_flight_pct_2d","label":"Frames sin apoyo detectado","value":flight_pct,"unit":"% dominio válido","quality":"Control interno","notes":"En marcha ordinaria un valor relevante sugiere fallo de segmentación; >5% invalida el doble apoyo."},
        {"key":"double_support_internal_discrepancy_pct","label":"Discrepancia interna de doble apoyo","value":ds_discrepancy,"unit":"puntos %","quality":"Control interno","notes":"Diferencia entre solapamiento observado y esperado por ocupación de apoyo; >10 puntos invalida el doble apoyo."},
        {"key":"double_support_time_2d","label":"Tiempo acumulado de doble apoyo","value":double_s,"unit":"s","quality":phase_quality,"notes":"Tiempo acumulado del segmento; no duración media por ciclo."},
        {"key":"swing_time_l_2d","label":"Tiempo de oscilación izquierdo estimado","value":swing_l,"unit":"s","quality":phase_quality,"notes":"TO izquierdo→siguiente IC izquierdo."},
        {"key":"swing_time_r_2d","label":"Tiempo de oscilación derecho estimado","value":swing_r,"unit":"s","quality":phase_quality,"notes":"TO derecho→siguiente IC derecho."},
        {"key":"swing_asymmetry_2d","label":"Asimetría de oscilación estimada","value":swing_asym,"unit":"%","quality":phase_quality,"notes":"Diferencia relativa D/I solo si la segmentación es consistente."},
        {"key":"initial_contacts_l_n","label":"Contactos iniciales izquierdos estimados","value":float(len(L["ic"])) if reliable else np.nan,"unit":"eventos","quality":phase_quality,"notes":"Eventos cinemáticos 2D estimados; no heel-strikes validados."},
        {"key":"initial_contacts_r_n","label":"Contactos iniciales derechos estimados","value":float(len(R["ic"])) if reliable else np.nan,"unit":"eventos","quality":phase_quality,"notes":"Eventos cinemáticos 2D estimados; no heel-strikes validados."},
    ]

    if not reliable:
        return out

    if is_frontal:
        for side, label, summary in [("L","izquierda",L),("R","derecha",R)]:
            ics = summary["ic"]
            foot = seg[f"{side}_foot_progress_proj"].to_numpy(float)
            rear = seg[f"{side}_rearfoot_tilt_proj"].to_numpy(float)
            valg = seg[f"{side}_dynamic_valgus"].to_numpy(float)
            out += [
                {"key":f"initial_contact_foot_{side.lower()}_deg","label":f"Orientación del pie {label} en contacto inicial estimado","value":_mean_at_indices(foot,ics),"unit":"°","quality":phase_quality,"notes":"Orientación distal 2D en IC estimado; no confirma talón/antepié."},
                {"key":f"initial_contact_rearfoot_{side.lower()}_deg","label":f"Retropié {label} en contacto inicial estimado","value":_mean_at_indices(rear,ics),"unit":"°","quality":phase_quality,"notes":"Inclinación proyectada del retropié en IC estimado."},
                {"key":f"loading_knee_{side.lower()}_deg","label":f"Desviación medial rodilla {label} en respuesta a la carga","value":_window_stat(valg,ics,None,0,10,"max"),"unit":"°","quality":phase_quality,"notes":"Máximo 2D en ventana 0–10% de ciclos aceptados; el valgo real es multiplanar."},
                {"key":f"terminal_foot_{side.lower()}_deg","label":f"Orientación distal pie {label} en pre-oscilación","value":_window_stat(foot,ics,None,50,60,"mean"),"unit":"°","quality":phase_quality,"notes":"Ventana 50–60% solo en ciclos aceptados."},
            ]
    else:
        for side, label, summary in [("L","izquierda",L),("R","derecha",R)]:
            ics = summary["ic"]
            ankle = seg[f"{side}_ankle_angle"].to_numpy(float)
            knee = seg[f"{side}_knee_flex"].to_numpy(float)
            out += [
                {"key":f"initial_contact_foot_{side.lower()}_deg","label":f"Ángulo tobillo-pie {label} en contacto inicial estimado","value":_mean_at_indices(ankle,ics),"unit":"°","quality":phase_quality,"notes":"Ángulo sagital 2D en IC estimado; no confirma heel-strike mediante fuerza."},
                {"key":f"loading_knee_{side.lower()}_deg","label":f"Flexión rodilla {label} en respuesta a la carga","value":_window_stat(knee,ics,None,0,10,"max"),"unit":"°","quality":phase_quality,"notes":"Máxima flexión proyectada en ventana 0–10% de ciclos aceptados."},
                {"key":f"terminal_foot_{side.lower()}_deg","label":f"Ángulo tobillo-pie {label} en pre-oscilación","value":_window_stat(ankle,ics,None,50,60,"mean"),"unit":"°","quality":phase_quality,"notes":"Ventana 50–60% de ciclos aceptados."},
            ]
    return out


def _robust_abs_p95(values):
    a=np.asarray(values,dtype=float)
    a=a[np.isfinite(a)]
    return float(np.nanpercentile(np.abs(a),95)) if len(a) else np.nan

def _signed_excursion_asymmetry_pct(values):
    """Asimetría entre excursión positiva y negativa (0=símétrica, 100=muy unilateral)."""
    a=np.asarray(values,dtype=float)
    a=a[np.isfinite(a)]
    if len(a)<10: return np.nan
    pos=max(0.0,float(np.nanpercentile(a,95)))
    neg=max(0.0,float(-np.nanpercentile(a,5)))
    den=max(pos,neg,1e-9)
    return float(np.clip(abs(pos-neg)/den*100.0,0.0,100.0))

def _cycle_shape_consistency_index(frames, values, event_frames, points=101):
    """Repetibilidad de la forma de la señal entre zancadas 2D. 100=curvas casi idénticas.
    Usa ciclos cada dos eventos alternantes y es un índice interno, no una escala validada.
    """
    fr=np.asarray(frames,dtype=float); y=np.asarray(values,dtype=float)
    ev=np.asarray(event_frames if event_frames is not None else [],dtype=float)
    ev=ev[np.isfinite(ev)]
    if len(fr)<8 or len(ev)<5: return np.nan
    cycles=[]
    xnew=np.linspace(0.0,1.0,points)
    for i in range(len(ev)-2):
        a,b=float(ev[i]),float(ev[i+2])
        mask=(fr>=a)&(fr<=b)&np.isfinite(y)
        if mask.sum()<6: continue
        yy=y[mask]; xx=fr[mask]
        if xx[-1]<=xx[0]: continue
        xn=(xx-xx[0])/(xx[-1]-xx[0])
        cycles.append(np.interp(xnew,xn,yy))
    if len(cycles)<3: return np.nan
    arr=np.asarray(cycles,float)
    med=np.nanmedian(arr,axis=0)
    amp=float(np.nanpercentile(arr,95)-np.nanpercentile(arr,5))
    if not np.isfinite(amp) or amp<1e-6: return 100.0
    rmse=np.sqrt(np.nanmean((arr-med[None,:])**2,axis=1))
    ratio=float(np.nanmedian(rmse)/amp)
    return float(np.clip(100.0-250.0*ratio,0.0,100.0))

def compute_metrics(df, fps, start_frame, end_frame, view, assistive_device="Sin ayuda", scale_cm_per_px=0.0):
    seg = df[(df.frame >= start_frame) & (df.frame <= end_frame)].copy()
    if len(seg) < max(30, int(fps * 2)):
        raise ValueError("El segmento seleccionado es demasiado corto.")
    seg = add_angle_columns(seg)
    is_frontal = "Frontal" in (view or "")
    if is_frontal:
        seg = add_frontal_columns(seg)
        seg = add_frontal_advanced(seg, fps, scale_cm_per_px)

    score_cols = [f"{n}_score" for n in LOWER_BODY]
    mean_tracking = float(seg[score_cols].mean(axis=1).mean())
    good_frames = float((seg[score_cols].min(axis=1) >= 0.5).mean() * 100)
    foot_visible = visibility_pct(seg, FOOT_POINTS, 0.5)
    upper_visible = visibility_pct(seg, UPPER_BODY, 0.5)
    q = quality_label(mean_tracking)
    assisted = assistive_device != "Sin ayuda"

    # v0.8.1 · dominio rectilíneo + ciclos IC→IC coherentes.
    # El giro y las transiciones no deben contaminar CV, asimetría ni doble apoyo.
    straight_mask, straight_info = _automatic_straight_walking_mask(seg, fps)
    seg["straight_walking_valid"] = straight_mask

    # V74 · también el QC publicado se calcula sobre el mismo dominio rectilíneo.
    # Se conserva la detección del giro sobre el segmento completo, pero una vez
    # definida la máscara no se mezclan frames de giro ni siquiera en visibilidad.
    _qc_seg = seg.loc[np.asarray(straight_mask, bool)].copy()
    if len(_qc_seg):
        mean_tracking = float(_qc_seg[score_cols].mean(axis=1).mean())
        good_frames = float((_qc_seg[score_cols].min(axis=1) >= 0.5).mean() * 100)
        foot_visible = visibility_pct(_qc_seg, FOOT_POINTS, 0.5)
        upper_visible = visibility_pct(_qc_seg, UPPER_BODY, 0.5)
        q = quality_label(mean_tracking)

    # Proxy distal se conserva solo para una primera estimación amplia del periodo.
    ly = (seg.LAnkle_y.to_numpy() + seg.LHeel_y.to_numpy() + seg.LBigToe_y.to_numpy()) / 3.0
    ry = (seg.RAnkle_y.to_numpy() + seg.RHeel_y.to_numpy() + seg.RBigToe_y.to_numpy()) / 3.0
    diff = rolling_smooth(ry - ly, 7)
    crossings = zero_crossings(diff)
    if len(crossings) > 1:
        kept = [crossings[0]]
        min_gap = max(1, int(round(0.25 * fps)))
        for c in crossings[1:]:
            if c - kept[-1] >= min_gap:
                kept.append(c)
        crossings = np.asarray(kept, dtype=int)

    # Periodo inicial robusto para estabilizar el detector de apoyo.
    intervals0 = np.diff(crossings) / fps if len(crossings) >= 2 else np.asarray([])
    mean_alt0 = float(np.nanmedian(intervals0)) if len(intervals0) >= 3 else np.nan
    expected_stride_s = (2.0 * mean_alt0) if np.isfinite(mean_alt0) and mean_alt0 > 0 else np.nan

    l_support, _, _ = _support_mask_2d(seg, "L", fps, expected_stride_s)
    r_support, _, _ = _support_mask_2d(seg, "R", fps, expected_stride_s)
    seg["L_support_2d"] = l_support
    seg["R_support_2d"] = r_support

    l_cycle0 = _support_cycle_summary(l_support, fps, expected_stride_s)
    r_cycle0 = _support_cycle_summary(r_support, fps, expected_stride_s)

    # Excluir ciclos que invaden giro/transiciones.
    l_cycle_orig = _filter_support_summary_to_mask(l_cycle0, straight_mask, fps, min_fraction=1.00)
    r_cycle_orig = _filter_support_summary_to_mask(r_cycle0, straight_mask, fps, min_fraction=1.00)

    # v0.10.9 · fallback robusto si la cadena original no supera QC temporal amplio.
    l_sel,r_sel,support_event_source,support_event_info = _select_temporal_event_chain(
        seg.reset_index(drop=True), fps, l_cycle_orig, r_cycle_orig
    )
    if support_event_source == "robust_distal":
        l_cycle = _filter_support_summary_to_mask(l_sel, straight_mask, fps, min_fraction=1.00)
        r_cycle = _filter_support_summary_to_mask(r_sel, straight_mask, fps, min_fraction=1.00)
    else:
        l_cycle,r_cycle = l_cycle_orig,r_cycle_orig

    support_reliable = l_cycle["quality"] in ("Alta", "Moderada") and r_cycle["quality"] in ("Alta", "Moderada")

    # v0.9.2 · Separación explícita de funciones:
    # - CV: robusto por lado (v0.9.1, preservado).
    # - Cadencia: secuencia anatómica L-R-L-R canónica.
    # - Cadencia ipsilateral: control secundario SIN filtro de outliers del CV.
    # - Cierre físico: apoyo L + apoyo R + doble apoyo observado.
    cycle_timing = _cycle_timing_metrics(l_cycle, r_cycle, fps) if support_reliable else {
        "cadence": np.nan, "mean_stride": np.nan, "cv": np.nan,
        "cv_left": np.nan, "cv_right": np.nan,
        "l_mean_stride": np.nan, "r_mean_stride": np.nan,
        "n_left": 0, "n_right": 0, "n_cycles": 0,
        "outliers_left": 0, "outliers_right": 0,
        "quality": "No fiable",
        "reason": "segmentación apoyo/oscilación insuficiente"
    }

    canonical = _canonical_gait_timeline(l_cycle, r_cycle, fps) if support_reliable else {
        "cadence": np.nan, "mean_step": np.nan, "asym": np.nan,
        "events": [], "n_intervals": 0, "quality": "No fiable",
        "reason": "segmentación apoyo/oscilación insuficiente"
    }

    cadence_steps = canonical.get("cadence", np.nan)
    cadence_stride, mean_stride_raw, n_stride_raw = _raw_stride_cadence(l_cycle, r_cycle) if support_reliable else (np.nan, np.nan, 0)

    # v0.10.6 · detector de ritmo completamente independiente del contacto.
    # Esto permite estimar cadencia/asimetría cuando el tracking articular es
    # excelente pero IC/TO o las máscaras de apoyo no cierran correctamente.
    kin_alt = _kinematic_alternation_metrics_straight_blocks(seg, fps, straight_mask)
    cadence_kin = kin_alt.get("cadence", np.nan)
    asym_kin = kin_alt.get("asym", np.nan)
    cv_kin = kin_alt.get("cv", np.nan)

    closure = _temporal_closure_from_masks(
        l_support, r_support, l_cycle, r_cycle, straight_mask, fps
    ) if support_reliable else {
        "double_raw": np.nan, "flight_pct": np.nan,
        "stance_pct_l": np.nan, "stance_pct_r": np.nan,
        "double_expected": np.nan, "double_discrepancy": np.nan,
        "closure_stride": np.nan, "closure_cadence": np.nan,
    }
    cadence_closure = closure.get("closure_cadence", np.nan)

    # CV permanece exactamente en la lógica robusta v0.9.1.
    cv_alt = cycle_timing.get("cv", np.nan)

    # Concordancias entre tres lecturas del MISMO fenómeno temporal.
    err_step_stride = _pct_disagreement(cadence_steps, cadence_stride)
    err_step_closure = _pct_disagreement(cadence_steps, cadence_closure)
    err_stride_closure = _pct_disagreement(cadence_stride, cadence_closure)

    available_cadences = [x for x in (cadence_steps, cadence_stride, cadence_closure) if np.isfinite(x)]
    pair_errors = [x for x in (err_step_stride, err_step_closure, err_stride_closure) if np.isfinite(x)]
    max_pair_error = float(max(pair_errors)) if pair_errors else np.nan

    # v0.10.6 · Jerarquía de cadencia:
    # A) pasos anatómicos IC L/R;
    # B) alternancia cinemática de pies (independiente de apoyo);
    # C) ciclos IC→IC ipsilaterales.
    #
    # El cierre apoyo/doble apoyo es QC, no requisito para contar pasos/min.
    n_step_intervals = int(canonical.get("n_intervals", 0) or 0)
    cadence_step_ok = np.isfinite(cadence_steps) and n_step_intervals >= 3
    cadence_kin_ok = np.isfinite(cadence_kin) and int(kin_alt.get("n_intervals",0)) >= 4
    cadence_stride_ok = np.isfinite(cadence_stride) and int(n_stride_raw) >= 2

    candidates_primary = []
    if cadence_step_ok:
        candidates_primary.append(("IC bilateral", float(cadence_steps)))
    if cadence_kin_ok:
        candidates_primary.append(("alternancia cinemática", float(cadence_kin)))
    if cadence_stride_ok:
        candidates_primary.append(("IC ipsilateral", float(cadence_stride)))

    cadence = np.nan
    cadence_quality = "No calculable"
    cadence_reason = "sin periodicidad distal suficiente"

    if cadence_step_ok and cadence_kin_ok:
        err = _pct_disagreement(cadence_steps, cadence_kin)
        if np.isfinite(err) and err <= 12.0:
            cadence = float(np.nanmean([cadence_steps, cadence_kin]))
            cadence_quality = "Alta" if err <= 7.0 else "Moderada"
            cadence_reason = f"IC bilateral y alternancia cinemática concuerdan ({err:.1f}%)"
        else:
            # Cuando las máscaras IC fallan pero el movimiento distal tiene varios
            # ciclos repetidos, preferir el detector cinemático y degradar calidad.
            cadence = float(cadence_kin)
            cadence_quality = "Moderada · detector cinemático"
            cadence_reason = (
                f"alternancia distal publicada; IC bilateral discrepa {err:.1f}%"
                if np.isfinite(err) else "alternancia distal con IC bilateral no comparable"
            )
    elif cadence_kin_ok:
        cadence = float(cadence_kin)
        cadence_quality = kin_alt.get("quality","Moderada") + " · detector cinemático"
        cadence_reason = kin_alt.get("reason","alternancia distal")
    elif cadence_step_ok:
        cadence = float(cadence_steps)
        cadence_quality = canonical.get("quality","Moderada")
        cadence_reason = canonical.get("reason","cadena anatómica IC")
    elif cadence_stride_ok:
        cadence = float(cadence_stride)
        cadence_quality = "Baja · ciclos ipsilaterales"
        cadence_reason = f"{n_stride_raw} ciclos IC→IC válidos"

    mean_alt = (60.0/cadence) if np.isfinite(cadence) and cadence > 0 else np.nan
    mean_stride_cycle = (120.0/cadence) if np.isfinite(cadence) and cadence > 0 else np.nan

    # Coherencia global sigue siendo estricta para métricas de CONTACTO.
    if len(available_cadences) >= 3:
        temporal_coherence_ok = bool(max_pair_error <= 10.0)
        temporal_coherence_quality = "Alta" if max_pair_error <= 6.0 else ("Moderada" if temporal_coherence_ok else "No fiable")
    elif len(available_cadences) == 2:
        temporal_coherence_ok = bool(max_pair_error <= 10.0)
        temporal_coherence_quality = "Moderada" if temporal_coherence_ok else "No fiable"
    else:
        temporal_coherence_ok = False
        temporal_coherence_quality = "No fiable"

    # Asimetría temporal: IC anatómico si el cierre es bueno; si no, usar
    # alternancia cinemática global, claramente etiquetada como tal.
    if temporal_coherence_ok and np.isfinite(canonical.get("asym", np.nan)):
        asym = canonical.get("asym", np.nan)
        asym_quality = temporal_coherence_quality
        asym_source = "IC anatómicos L→R/R→L"
    elif np.isfinite(asym_kin) and int(kin_alt.get("n_intervals",0)) >= 4:
        asym = float(asym_kin)
        asym_quality = kin_alt.get("quality","Baja") + " · cinemática"
        asym_source = "alternancia distal P→T/T→P"
    else:
        asym = np.nan
        asym_quality = "No calculable"
        asym_source = "sin eventos temporales suficientes"

    # CV: conservar el de ciclos por lado cuando existe; si no, usar el CV del
    # intervalo de alternancia distal para no dejar el registro vacío.
    if not np.isfinite(cv_alt) and np.isfinite(cv_kin):
        cv_alt = float(cv_kin)
        cv_quality_override = kin_alt.get("quality","Baja") + " · cinemática"
    else:
        cv_quality_override = None

    if support_reliable:
        stance_l = float(np.nanmean(l_cycle["stance"])) if len(l_cycle["stance"]) else np.nan
        stance_r = float(np.nanmean(r_cycle["stance"])) if len(r_cycle["stance"]) else np.nan
    else:
        stance_l = stance_r = np.nan

    stance_asym = abs(stance_l-stance_r)/((stance_l+stance_r)/2.0)*100.0 if np.isfinite(stance_l) and np.isfinite(stance_r) and (stance_l+stance_r)>0 else np.nan
    stance_longer = 1.0 if np.isfinite(stance_l) and np.isfinite(stance_r) and stance_l>stance_r else (2.0 if np.isfinite(stance_l) and np.isfinite(stance_r) and stance_r>stance_l else 0.0)

    segment_duration_s = float((seg.frame.max() - seg.frame.min() + 1) / fps)
    valid_duration_s = float(np.sum(straight_mask) / fps)

    detected_steps = int(len(canonical.get("events", [])))
    cadence_count_segment = (60.0 * detected_steps / valid_duration_s) if valid_duration_s > 0 and detected_steps > 0 else np.nan
    expected_steps = (cadence * valid_duration_s / 60.0) if np.isfinite(cadence) else np.nan
    consistency_error = (
        abs(detected_steps - expected_steps) / max(expected_steps, 1.0) * 100.0
        if np.isfinite(expected_steps) else np.nan
    )

    if not np.isfinite(cadence):
        consistency_quality = "Revisar"
    elif detected_steps < 4 or not np.isfinite(consistency_error):
        consistency_quality = "Limitada"
    elif consistency_error <= 10:
        consistency_quality = "Alta"
    elif consistency_error <= 20:
        consistency_quality = "Moderada"
    else:
        consistency_quality = "Revisar"

    # v25 · Consistencia paso a paso: cuánto se parece cada intervalo al inmediatamente anterior.
    _step_intervals = np.asarray(kin_alt.get("intervals", []), dtype=float)
    _step_intervals = _step_intervals[np.isfinite(_step_intervals) & (_step_intervals > 0)]
    if len(_step_intervals) >= 3:
        _pair_den = (_step_intervals[1:] + _step_intervals[:-1]) / 2.0
        _pair_change = np.divide(np.abs(np.diff(_step_intervals)), _pair_den, out=np.full(len(_pair_den), np.nan), where=_pair_den>1e-9) * 100.0
        step_to_step_change = float(np.nanmedian(_pair_change)) if np.isfinite(_pair_change).any() else np.nan
        step_to_step_similarity = float(np.clip(100.0 - step_to_step_change, 0.0, 100.0)) if np.isfinite(step_to_step_change) else np.nan
        step_consistency_quality = kin_alt.get("quality", "Baja")
    else:
        step_to_step_change = step_to_step_similarity = np.nan
        step_consistency_quality = "No calculable"

    aid_note = f" Marcha con ayuda técnica: {assistive_device}; revisar oclusiones." if assisted else ""
    metrics = [
        {"key":"support_event_source","label":"Fuente de eventos temporales","value":support_event_source,"unit":"","quality":support_event_info.get("quality","") if isinstance(support_event_info,dict) else "","notes":support_event_info.get("reason","") if isinstance(support_event_info,dict) else ""},
        {"key":"support_event_quality","label":"Calidad cadena IC/TO alternativa","value":support_event_info.get("quality","") if isinstance(support_event_info,dict) else "","unit":"","quality":support_event_info.get("quality","") if isinstance(support_event_info,dict) else "","notes":"QC interno; no equivale a validación con plataforma de fuerza."},
        {"key":"tracking_mean","label":"Confianza media del tracking","value":mean_tracking,"unit":"","quality":q,"notes":"Media HALPE26 del tren inferior."+aid_note},
        {"key":"good_frames_pct","label":"Frames con tren inferior visible ≥0,50","value":good_frames,"unit":"%","quality":q,"notes":"Puntos principales del tren inferior ≥0,50."+aid_note},
        {"key":"foot_visibility_pct","label":"Visibilidad de pie/tobillo","value":foot_visible,"unit":"%","quality":quality_label((foot_visible or 0)/100),"notes":"Tobillo, talón y antepié. Fundamental para métricas del pie."+aid_note},
        {"key":"upper_visibility_pct","label":"Visibilidad del tren superior","value":upper_visible,"unit":"%","quality":quality_label((upper_visible or 0)/100),"notes":"Hombros, codos y muñecas; puede disminuir con muletas/caminador."+aid_note},
        {"key":"straight_walking_usable_pct","label":"Tramo rectilíneo utilizable","value":float(straight_info["usable_pct"]),"unit":"% segmento","quality":"Control interno","notes":straight_info["reason"]+"; V74 aplica esta máscara a métricas temporales y espaciales, perfiles por fase, gráficas e informe."},
        {"key":"turn_transition_excluded_pct","label":"Giro/transiciones excluidos","value":float(straight_info["excluded_pct"]),"unit":"% segmento","quality":"Control interno","notes":"Exclusión automática por orientación corporal/inversión de trayectoria; no es una métrica clínica."},
        {"key":"straight_walking_blocks","label":"Bloques rectilíneos válidos","value":float(len([x for x in np.unique(_straight_block_ids(straight_mask)) if x>0])),"unit":"bloques","quality":"Control interno","notes":"V74: cada bloque continuo se mantiene separado; los detectores de periodicidad no interpolan a través del giro."},
        {"key":"straight_mask_global_flag","label":"Máscara rectilínea global V74","value":1.0,"unit":"bool","quality":"Activa","notes":"La misma máscara gobierna métricas, perfiles por fase, gráficas, biplanar e informe; las fórmulas biomecánicas no se modifican."},
        {"key":"cadence_exp","label":"Cadencia estimada","value":cadence,"unit":"pasos/min","quality":cadence_quality,"notes":"v0.10.6: jerarquía IC bilateral + alternancia cinemática distal + ciclos IC→IC. El cierre de apoyo no impide contar pasos/min si la periodicidad articular es suficiente. "+cadence_reason},
        {"key":"cadence_kinematic","label":"Cadencia por alternancia cinemática","value":cadence_kin,"unit":"pasos/min","quality":kin_alt.get("quality","No fiable"),"notes":"Estimación independiente de las máscaras de apoyo a partir de máximos/mínimos alternantes de la trayectoria relativa de ambos pies."},
        {"key":"kinematic_alternation_events","label":"Eventos de alternancia cinemática","value":float(len(kin_alt.get("event_frames",[]))),"unit":"eventos","quality":kin_alt.get("quality","No fiable"),"notes":"Extremos alternantes del movimiento distal usados como detector de periodicidad; no equivalen a heel-strikes de plataforma."},
        {"key":"kinematic_periodicity","label":"Periodicidad cinemática","value":kin_alt.get("periodicity",np.nan),"unit":"r autocorr","quality":kin_alt.get("quality","No fiable"),"notes":"Pico de autocorrelación de la señal distal detrendida; control interno de repetición."},
        {"key":"step_events_detected","label":"Contactos IC anatómicos en cadena válida","value":float(detected_steps),"unit":"eventos","quality":canonical.get("quality","No fiable"),"notes":"Contactos iniciales L/R derivados de ciclos de apoyo y ordenados en una única secuencia alternante; no son heel-strikes de plataforma de fuerzas."},
        {"key":"segment_duration_s","label":"Duración del segmento analizado","value":segment_duration_s,"unit":"s","quality":"Directa","notes":"Duración temporal total del intervalo seleccionado."},
        {"key":"valid_straight_duration_s","label":"Duración rectilínea utilizable","value":valid_duration_s,"unit":"s","quality":"Control interno","notes":"Tiempo realmente usado tras excluir bordes/giro/transiciones."},
        {"key":"stride_cycle_duration_s","label":"Duración de zancada desde cadena alternante","value":mean_stride_cycle,"unit":"s","quality":temporal_coherence_quality if np.isfinite(mean_stride_cycle) else "No fiable","notes":"Dos intervalos de paso de la línea temporal anatómica v0.9.2; se suprime si falla la coherencia temporal."},
        {"key":"cadence_count_segment","label":"Cadencia por recuento/duración rectilínea","value":cadence_count_segment,"unit":"pasos/min","quality":"Control interno" if np.isfinite(cadence_count_segment) else "No calculable","notes":"60 × contactos aceptados / duración rectilínea utilizable; solo control secundario."},
        {"key":"expected_steps_from_cadence","label":"Eventos esperados por cadencia × duración válida","value":expected_steps,"unit":"eventos","quality":"Control interno" if np.isfinite(expected_steps) else "No calculable","notes":"Comprobación interna sobre el dominio rectilíneo: cadencia × duración válida / 60."},
        {"key":"cadence_contact_crosscheck_error_pct","label":"Discrepancia cadencia pasos vs ciclos ipsilaterales","value":err_step_stride,"unit":"%","quality":"Alta" if np.isfinite(err_step_stride) and err_step_stride<=10 else ("Revisar" if np.isfinite(err_step_stride) else "No calculable"),"notes":"Control v0.9.2 entre la cadena L-R anatómica y los ciclos IC→IC ipsilaterales sin filtrado del CV."},
        {"key":"cadence_closure_crosscheck_error_pct","label":"Discrepancia cadencia pasos vs cierre apoyo/doble apoyo","value":err_step_closure,"unit":"%","quality":"Alta" if np.isfinite(err_step_closure) and err_step_closure<=10 else ("Revisar" if np.isfinite(err_step_closure) else "No calculable"),"notes":"Contrasta la cadencia de pasos con la cadencia físicamente compatible con apoyo izquierdo + apoyo derecho + doble apoyo observado."},
        {"key":"cadence_stride_closure_error_pct","label":"Discrepancia ciclos vs cierre físico","value":err_stride_closure,"unit":"%","quality":"Alta" if np.isfinite(err_stride_closure) and err_stride_closure<=10 else ("Revisar" if np.isfinite(err_stride_closure) else "No calculable"),"notes":"Tercer control independiente de coherencia temporal."},
        {"key":"temporal_coherence_max_error_pct","label":"Discordancia temporal máxima","value":max_pair_error,"unit":"%","quality":temporal_coherence_quality,"notes":"Máxima discrepancia entre cadencia por pasos anatómicos, ciclos ipsilaterales y cierre apoyo/doble apoyo. Umbral operativo de publicación: ≤10%."},
        {"key":"temporal_coherence_flag","label":"Coherencia temporal global","value":1.0 if temporal_coherence_ok else 0.0,"unit":"bool","quality":temporal_coherence_quality,"notes":"1 = cadencia/asimetría/doble apoyo superan el control de coherencia v0.9.2; 0 = se suprimen las métricas incompatibles."},
        {"key":"cadence_candidate_steps","label":"Cadencia candidata por pasos anatómicos","value":cadence_steps,"unit":"pasos/min","quality":"Control interno","notes":"No se interpreta aisladamente; candidato primario antes del cierre de coherencia."},
        {"key":"cadence_candidate_stride","label":"Cadencia candidata por ciclos ipsilaterales","value":cadence_stride,"unit":"pasos/min","quality":"Control interno","notes":"120 / duración media de ciclos IC→IC plausibles, sin aplicar el filtro estadístico usado para CV."},
        {"key":"cadence_candidate_closure","label":"Cadencia candidata por cierre apoyo/doble apoyo","value":cadence_closure,"unit":"pasos/min","quality":"Control interno","notes":"120 / [(apoyo_L + apoyo_R)/(1 + doble_apoyo_fracción)], solo si no existe fase aérea relevante."},
        {"key":"step_count_consistency_error_pct","label":"Discrepancia recuento-cadencia-duración","value":consistency_error,"unit":"%","quality":consistency_quality,"notes":"Diferencia relativa entre eventos detectados y eventos esperados por cadencia × duración. Sirve como control de consistencia, no como validación clínica."},
        {"key":"alternation_interval","label":"Intervalo medio de paso anatómico","value":mean_alt,"unit":"s","quality":temporal_coherence_quality if np.isfinite(mean_alt) else "No calculable","notes":"IC-L→IC-R o IC-R→IC-L de la cadena anatómica v0.9.2."},
        {"key":"regularity_cv","label":"Variabilidad temporal global robusta (CV)","value":cv_alt,"unit":"%","quality":(cv_quality_override if cv_quality_override else cycle_timing.get("quality","No fiable")) if np.isfinite(cv_alt) else "No calculable","notes":"v0.10.6: usa CV por ciclos IC→IC cuando es válido; si esos ciclos fallan pero la alternancia distal es repetitiva, informa el CV de intervalos cinemáticos con calidad explícita."},
        {"key":"step_to_step_similarity_index","label":"Índice de consistencia paso a paso","value":step_to_step_similarity,"unit":"/100","quality":step_consistency_quality,"notes":"v25: repetibilidad temporal entre cada intervalo de paso y el inmediatamente anterior. 100=máxima similitud temporal. Índice interno experimental; no es una escala clínica validada."},
        {"key":"step_to_step_change_pct","label":"Cambio mediano entre pasos consecutivos","value":step_to_step_change,"unit":"%","quality":step_consistency_quality,"notes":"v25: diferencia porcentual robusta entre intervalos consecutivos de alternancia distal. Menor valor = pasos temporalmente más parecidos."},
        {"key":"regularity_cv_left","label":"CV temporal izquierdo","value":cycle_timing.get("cv_left"),"unit":"%","quality":cycle_timing.get("quality","No fiable"),"notes":"CV de duración IC→IC de la extremidad izquierda tras rechazo robusto de ciclos atípicos."},
        {"key":"regularity_cv_right","label":"CV temporal derecho","value":cycle_timing.get("cv_right"),"unit":"%","quality":cycle_timing.get("quality","No fiable"),"notes":"CV de duración IC→IC de la extremidad derecha tras rechazo robusto de ciclos atípicos."},
        {"key":"valid_cycles_left","label":"Ciclos válidos izquierdos","value":cycle_timing.get("n_left"),"unit":"ciclos","quality":"Control interno","notes":"Número de ciclos IC→IC izquierdos usados para cadencia/CV."},
        {"key":"valid_cycles_right","label":"Ciclos válidos derechos","value":cycle_timing.get("n_right"),"unit":"ciclos","quality":"Control interno","notes":"Número de ciclos IC→IC derechos usados para cadencia/CV."},
        {"key":"cycle_outliers_left","label":"Ciclos atípicos excluidos izquierdos","value":cycle_timing.get("outliers_left"),"unit":"ciclos","quality":"Control interno","notes":"Ciclos descartados por filtro robusto MAD/mediana."},
        {"key":"cycle_outliers_right","label":"Ciclos atípicos excluidos derechos","value":cycle_timing.get("outliers_right"),"unit":"ciclos","quality":"Control interno","notes":"Ciclos descartados por filtro robusto MAD/mediana."},
        {"key":"temporal_asymmetry_exp","label":"Asimetría temporal global","value":asym,"unit":"%","quality":asym_quality,"notes":"v0.10.6: fuente="+asym_source+". Si los IC anatómicos no son fiables, puede utilizar alternancia distal cinemática; esta describe asimetría temporal de alternancia y no contacto de fuerzas."},
        {"key":"support_segmentation_score_l","label":"Consistencia segmentación apoyo izquierdo","value":float(l_cycle["score"]),"unit":"/100","quality":l_cycle["quality"],"notes":l_cycle["reason"]+"; control matemático, no índice clínico."},
        {"key":"support_segmentation_score_r","label":"Consistencia segmentación apoyo derecho","value":float(r_cycle["score"]),"unit":"/100","quality":r_cycle["quality"],"notes":r_cycle["reason"]+"; control matemático, no índice clínico."},
        {"key":"stance_time_l_2d","label":"Tiempo de apoyo izquierdo estimado 2D","value":stance_l,"unit":"s","quality":"Experimental · ciclo continuo" if np.isfinite(stance_l) else "No fiable","notes":"IC→TO dentro de ciclos continuos estabilizados; se anula si la segmentación no supera el control temporal."},
        {"key":"stance_time_r_2d","label":"Tiempo de apoyo derecho estimado 2D","value":stance_r,"unit":"s","quality":"Experimental · ciclo continuo" if np.isfinite(stance_r) else "No fiable","notes":"IC→TO dentro de ciclos continuos estabilizados; se anula si la segmentación no supera el control temporal."},
        {"key":"stance_asymmetry_2d","label":"Asimetría del tiempo de apoyo estimado 2D","value":stance_asym,"unit":"%","quality":"Experimental · ciclo continuo" if np.isfinite(stance_asym) else "No fiable","notes":"Solo se calcula si ambos lados superan el control de consistencia de ciclo."},
        {"key":"stance_longer_side_code","label":"Extremidad con mayor apoyo estimado (código)","value":stance_longer,"unit":"1=I,2=D","quality":"Experimental" if stance_longer else "No fiable","notes":"1=izquierda; 2=derecha. No equivale a medición con plataforma/plantilla instrumentada."},
    ]
    # V74 · A partir de aquí TODA la biomecánica espacial, perfiles por fase,
    # gráficas, biplanar e informe reciben el mismo segmento enmascarado.
    # Las máscaras de apoyo se conservan, pero quedan forzadas a False fuera
    # del dominio rectilíneo para impedir que un giro llegue a Trendelenburg/BOS.
    seg = _mask_seg_to_straight_domain(seg, straight_mask)
    if "L_support_2d" in seg.columns:
        seg["L_support_2d"] = seg["L_support_2d"].fillna(False).astype(bool) & straight_mask
    if "R_support_2d" in seg.columns:
        seg["R_support_2d"] = seg["R_support_2d"].fillna(False).astype(bool) & straight_mask

    metrics += compute_gait_phase_metrics(
        seg, fps, is_frontal,
        expected_stride_s=expected_stride_s,
        support_summary_l=l_cycle,
        support_summary_r=r_cycle,
        temporal_coherence_ok=temporal_coherence_ok,
    )

    if not is_frontal:
        for key_base, label_base, col_base in [
            ("knee_flex", "Flexión rodilla", "knee_flex"),
            ("hip_flex", "Flexión cadera", "hip_flex"),
            ("ankle_angle", "Ángulo tobillo-pie", "ankle_angle"),
            ("shoulder_elev", "Elevación hombro", "shoulder_elev"),
        ]:
            vals = {}
            for side, side_name in [("L","izquierda"),("R","derecha")]:
                arr = seg[f"{side}_{col_base}"].to_numpy(dtype=float)
                p95 = float(np.nanpercentile(arr, 95)); rom = robust_rom(arr); vals[side] = p95
                quality = "Condicionada por ayuda técnica" if assisted and key_base=="shoulder_elev" else q
                metrics += [
                    {"key":f"{key_base}_{side.lower()}_p95","label":f"{label_base} {side_name} 2D (P95)","value":p95,"unit":"°","quality":quality,"notes":f"Ángulo 2D proyectado en vista {view}."+aid_note},
                    {"key":f"{key_base}_{side.lower()}_rom","label":f"ROM {label_base.lower()} {side_name} 2D","value":rom,"unit":"°","quality":quality,"notes":"ROM robusto P95-P5; 2D proyectado."+aid_note},
                ]
            metrics.append({"key":f"{key_base}_diff_p95","label":f"Diferencia D/I {label_base.lower()} 2D","value":abs(vals["L"]-vals["R"]),"unit":"°","quality":q,"notes":"Diferencia absoluta P95 (percentil 95) D/I; 2D proyectado."})
    else:
        vals = {}
        for side, side_name in [("L","izquierda"),("R","derecha")]:
            kd = seg[f"{side}_frontal_knee_dev"].to_numpy(float)
            fp = seg[f"{side}_foot_progress_proj"].to_numpy(float)
            rf = seg[f"{side}_rearfoot_tilt_proj"].to_numpy(float)
            vals[side] = {
                "knee": float(np.nanpercentile(kd,95)),
                "foot": float(np.nanmedian(fp)),
                "rear": float(np.nanpercentile(np.abs(rf),95)),
            }
            foot_q = q if foot_visible >= 80 else "Baja/condicionada"
            metrics += [
                {"key":f"frontal_knee_dev_{side.lower()}_p95","label":f"Desviación frontal rodilla {side_name} (P95)","value":vals[side]["knee"],"unit":"°","quality":q,"notes":"Magnitud proyectada del eje cadera-rodilla-tobillo. No diagnostica valgo/varo 3D."},
                {"key":f"foot_progress_{side.lower()}_median","label":f"Orientación del pie {side_name} proyectada (mediana)","value":vals[side]["foot"],"unit":"°","quality":foot_q,"notes":"Proxy distal de orientación/rotación en la imagen. No equivale a rotación axial de cadera."},
                {"key":f"rearfoot_tilt_{side.lower()}_p95","label":f"Inclinación retropié {side_name} proyectada (P95 (percentil 95) abs.)","value":vals[side]["rear"],"unit":"°","quality":foot_q,"notes":"Eje tobillo-talón proyectado. Puede sugerir cambios de eversión/inversión, pero no mide pronación 3D."},
            ]
        metrics += [
            {"key":"frontal_knee_dev_diff","label":"Diferencia D/I desviación frontal de rodilla","value":abs(vals['L']['knee']-vals['R']['knee']),"unit":"°","quality":q,"notes":"Comparación 2D proyectada."},
            {"key":"foot_progress_diff","label":"Diferencia D/I orientación del pie proyectada","value":abs(vals['L']['foot']-vals['R']['foot']),"unit":"°","quality":q,"notes":"Proxy distal; no atribuir directamente a rotación de cadera."},
            {"key":"rearfoot_tilt_diff","label":"Diferencia D/I inclinación del retropié proyectada","value":abs(vals['L']['rear']-vals['R']['rear']),"unit":"°","quality":q,"notes":"No equivale a pronación clínica."},
            {"key":"pelvis_obliquity_rom","label":"ROM oblicuidad pélvica proyectada","value":robust_rom(seg.pelvis_obliquity),"unit":"°","quality":q,"notes":"P95-P5 de la línea inter-caderas en el plano de imagen."},
            {"key":"shoulder_obliquity_rom","label":"ROM oblicuidad de hombros proyectada","value":robust_rom(seg.shoulder_obliquity),"unit":"°","quality":"Condicionada por ayuda técnica" if assisted else q,"notes":"P95-P5 de la línea biacromial en el plano de imagen."+aid_note},
            {"key":"trunk_lateral_lean_rom","label":"ROM (Rango de Movimiento) inclinación lateral del tronco proyectada","value":robust_rom(seg.trunk_lateral_lean),"unit":"°","quality":q,"notes":"P95-P5 del eje centro de pelvis-centro de hombros respecto a la vertical de la imagen."},
            {"key":"shoulder_pelvis_rel_rom","label":"ROM relación hombros-pelvis proyectada","value":robust_rom(seg.shoulder_pelvis_rel),"unit":"°","quality":"Condicionada por ayuda técnica" if assisted else q,"notes":"Variación de la diferencia entre oblicuidad de hombros y pelvis; descriptor 2D del acoplamiento tronco-pelvis."+aid_note},
            {"key":"base_width_relative_median","label":"Anchura de base relativa (tobillos/pelvis)","value":float(np.nanmedian(seg.base_width_relative)),"unit":"ratio","quality":q,"notes":"Anchura proyectada normalizada por anchura pélvica; no es distancia métrica sin calibración."},
        ]
        # Parámetros frontales avanzados, condicionados por tracking y escala espacial.
        com_amp_px = robust_rom(seg.com_proxy_x)
        com_cm = robust_rom(seg.com_proxy_cm) if np.isfinite(seg.com_proxy_cm.to_numpy(float)).any() else np.nan
        com_norm = robust_rom(seg.com_proxy_norm_pct_pelvis) if np.isfinite(seg.com_proxy_norm_pct_pelvis.to_numpy(float)).any() else np.nan
        com_vel = float(np.nanpercentile(np.abs(seg.com_velocity_px_s.to_numpy(float)), 95)) if np.isfinite(seg.com_velocity_px_s.to_numpy(float)).any() else np.nan
        com_bos_p95 = float(np.nanpercentile(np.abs(seg.com_bos_ratio.to_numpy(float)), 95)) if np.isfinite(seg.com_bos_ratio.to_numpy(float)).any() else np.nan
        com_bos_mean = float(np.nanmean(np.abs(seg.com_bos_ratio.to_numpy(float)))) if np.isfinite(seg.com_bos_ratio.to_numpy(float)).any() else np.nan
        if support_reliable:
            ds = seg[seg.L_support_2d & seg.R_support_2d]
            bos_cm = float(np.nanmedian(ds.bos_cm)) if len(ds)>=3 and np.isfinite(ds.bos_cm.to_numpy(float)).any() else np.nan
            l_single = seg[seg.L_support_2d & ~seg.R_support_2d]
            r_single = seg[seg.R_support_2d & ~seg.L_support_2d]
            # Convención de pelvis: + = lado izquierdo elevado / derecho descendido; - = lado izquierdo descendido / derecho elevado.
            trend_r_swing = float(max(0.0, np.nanpercentile(l_single.pelvis_obliquity,95))) if len(l_single)>=3 else np.nan
            trend_l_swing = float(max(0.0, -np.nanpercentile(r_single.pelvis_obliquity,5))) if len(r_single)>=3 else np.nan
        else:
            bos_cm = np.nan
            trend_r_swing = trend_l_swing = np.nan
        valg_l = float(np.nanpercentile(seg.L_dynamic_valgus,95)); valg_r = float(np.nanpercentile(seg.R_dynamic_valgus,95))
        # v29 · Perfil frontal: magnitud postural, balance lateral y repetibilidad entre ciclos.
        _fr=seg.frame.to_numpy(float)
        _ev=kin_alt.get("event_frames",[])
        pelvis_abs95=_robust_abs_p95(seg.pelvis_obliquity)
        pelvis_side_asym=_signed_excursion_asymmetry_pct(seg.pelvis_obliquity)
        pelvis_cycle_cons=_cycle_shape_consistency_index(_fr,seg.pelvis_obliquity,_ev)
        trunk_abs95=_robust_abs_p95(seg.trunk_lateral_lean)
        trunk_side_asym=_signed_excursion_asymmetry_pct(seg.trunk_lateral_lean)
        trunk_cycle_cons=_cycle_shape_consistency_index(_fr,seg.trunk_lateral_lean,_ev)
        shoulder_abs95=_robust_abs_p95(seg.shoulder_obliquity)
        shoulder_side_asym=_signed_excursion_asymmetry_pct(seg.shoulder_obliquity)
        shoulder_cycle_cons=_cycle_shape_consistency_index(_fr,seg.shoulder_obliquity,_ev)
        shoulder_pelvis_abs95=_robust_abs_p95(seg.shoulder_pelvis_rel)
        shoulder_pelvis_cycle_cons=_cycle_shape_consistency_index(_fr,seg.shoulder_pelvis_rel,_ev)
        com_side_asym=_signed_excursion_asymmetry_pct(seg.com_proxy_norm_pct_pelvis)
        com_cycle_cons=_cycle_shape_consistency_index(_fr,seg.com_proxy_norm_pct_pelvis,_ev)
        com_bos_cycle_cons=_cycle_shape_consistency_index(_fr,seg.com_bos_ratio,_ev)

        # Acoplamiento tronco-pelvis V74: misma fórmula, aplicada POR BLOQUES
        # rectilíneos. Nunca se interpola a través del cambio de sentido.
        _block_ids = _straight_block_ids(straight_mask)
        _couplings=[]; _phases=[]; _weights=[]
        for _bid in [x for x in np.unique(_block_ids) if x>0]:
            _idx=np.where(_block_ids==_bid)[0]
            if len(_idx)<=10:
                continue
            _a0=pd.to_numeric(seg.trunk_lateral_lean.iloc[_idx],errors="coerce")
            _b0=pd.to_numeric(seg.pelvis_obliquity.iloc[_idx],errors="coerce")
            _ok=np.isfinite(_a0.to_numpy(float)) & np.isfinite(_b0.to_numpy(float))
            if int(_ok.sum())<=10:
                continue
            # Interpolación permitida únicamente DENTRO del bloque rectilíneo.
            a=_a0.interpolate(limit_direction="both").to_numpy(float)
            b=_b0.interpolate(limit_direction="both").to_numpy(float)
            if len(a)>10 and np.nanstd(a)>1e-6 and np.nanstd(b)>1e-6:
                _r=float(np.corrcoef(a,b)[0,1])
                if np.isfinite(_r):
                    _couplings.append(_r); _weights.append(len(a))
            if len(a)>20 and np.isfinite(mean_alt) and mean_alt>0:
                aa=(a-np.nanmean(a))/(np.nanstd(a)+1e-8); bb=(b-np.nanmean(b))/(np.nanstd(b)+1e-8)
                maxlag=min(int(round(fps)),len(a)//3); bestlag=0; best=-np.inf
                for lag in range(-maxlag,maxlag+1):
                    x,y=(aa[:-lag],bb[lag:]) if lag>0 else ((aa[-lag:],bb[:lag]) if lag<0 else (aa,bb))
                    if len(x)>10:
                        c=np.corrcoef(x,y)[0,1]
                        if np.isfinite(c) and abs(c)>best: best=abs(c); bestlag=lag
                stride_period = mean_stride_cycle if np.isfinite(mean_stride_cycle) and mean_stride_cycle > 0 else (2.0 * mean_alt)
                _phase=float(bestlag / fps / stride_period * 360.0)
                _phase=float(((_phase + 180.0) % 360.0) - 180.0)
                if np.isfinite(_phase): _phases.append((_phase,len(a)))
        coupling_r=float(np.average(_couplings,weights=_weights[:len(_couplings)])) if _couplings else np.nan
        if _phases:
            # Media circular ponderada para no promediar mal +179/-179.
            _ang=np.deg2rad([x[0] for x in _phases]); _w=np.asarray([x[1] for x in _phases],float)
            phase_deg=float(np.degrees(np.arctan2(np.sum(_w*np.sin(_ang)),np.sum(_w*np.cos(_ang)))))
        else:
            phase_deg=np.nan
        metrics += [
            {"key":"com_lateral_excursion_cm","label":"Oscilación lateral CoM proxy 2D","value":com_cm,"unit":"cm","quality":q if np.isfinite(com_cm) else "Requiere escala","notes":"Proxy tronco-pelvis. Se expresa en cm solo con escala espacial 2D calibrada; no es CoM segmentario 3D."},
            {"key":"com_lateral_excursion_px","label":"Oscilación lateral CoM proxy 2D (imagen)","value":com_amp_px,"unit":"px","quality":q,"notes":"Amplitud robusta P95-P5 en imagen; útil para seguimiento si la cámara permanece idéntica."},
            {"key":"com_lateral_excursion_norm_pct","label":"Excursión mediolateral COM (Centro de Masa estimado 2D) normalizada","value":com_norm,"unit":"% anchura pélvica","quality":q if np.isfinite(com_norm) else "No calculable","notes":"v25: P95-P5 del COM proxy, normalizado por la anchura pélvica aparente. Facilita comparación intraindividual; no es COM 3D."},
            {"key":"com_velocity_ml_px_s","label":"Velocidad mediolateral COM (Centro de Masa estimado 2D) (P95)","value":com_vel,"unit":"px/s","quality":q if np.isfinite(com_vel) else "No calculable","notes":"v25: P95 (percentil 95) de la velocidad lateral absoluta del COM proxy. Dependiente de la geometría de cámara."},
            {"key":"com_bos_ratio_p95","label":"COM (Centro de Masa estimado 2D) / BOS (Base de Sustentación) lateral máximo robusto","value":com_bos_p95,"unit":"ratio","quality":q if np.isfinite(com_bos_p95) else "No calculable","notes":"v25: distancia lateral absoluta del COM proxy respecto al centro entre ambos pies, dividida por media BOS. 0=centro; ~1=centro de un pie."},
            {"key":"com_bos_ratio_mean_abs","label":"COM (Centro de Masa estimado 2D) / BOS (Base de Sustentación) lateral medio","value":com_bos_mean,"unit":"ratio","quality":q if np.isfinite(com_bos_mean) else "No calculable","notes":"v25: alejamiento lateral medio del COM proxy respecto al centro de la base estimada."},
            {"key":"bos_width_cm","label":"Ancho de base de sustentación estimado","value":bos_cm,"unit":"cm","quality":q if np.isfinite(bos_cm) else "Requiere escala/apoyo","notes":"Mediana entre centros de ambos pies durante doble apoyo 2D estimado."},
            {"key":"trendelenburg_drop_l_deg","label":"Caída pélvica dinámica lado izquierdo en suspensión","value":trend_l_swing,"unit":"°","quality":"Experimental" if support_reliable and np.isfinite(trend_l_swing) else "No fiable","notes":"Ángulo proyectado durante apoyo derecho; se anula si la segmentación temporal de apoyo no es consistente. No constituye por sí solo diagnóstico de Trendelenburg."},
            {"key":"trendelenburg_drop_r_deg","label":"Caída pélvica dinámica lado derecho en suspensión","value":trend_r_swing,"unit":"°","quality":"Experimental" if support_reliable and np.isfinite(trend_r_swing) else "No fiable","notes":"Ángulo proyectado durante apoyo izquierdo; se anula si la segmentación temporal de apoyo no es consistente. No constituye por sí solo diagnóstico de Trendelenburg."},
            {"key":"dynamic_knee_valgus_l_deg","label":"Valgo dinámico proyectado rodilla izquierda (P95)","value":valg_l,"unit":"°","quality":q,"notes":"Desviación medial proyectada respecto al eje cadera-tobillo; el valgo real es multiplanar."},
            {"key":"dynamic_knee_valgus_r_deg","label":"Valgo dinámico proyectado rodilla derecha (P95)","value":valg_r,"unit":"°","quality":q,"notes":"Desviación medial proyectada respecto al eje cadera-tobillo; el valgo real es multiplanar."},
            {"key":"pelvis_obliquity_p95_abs","label":"Oblicuidad pélvica absoluta robusta","value":pelvis_abs95,"unit":"°","quality":q,"notes":"Versión 30: P95 (percentil 95) del valor absoluto. Detecta también posturas sostenidas que un ROM pequeño podría ocultar; proxy 2D."},
            {"key":"pelvis_obliquity_side_asymmetry_pct","label":"Asimetría de excursión pélvica lateral","value":pelvis_side_asym,"unit":"%","quality":q,"notes":"Versión 30: diferencia relativa entre excursiones positivas y negativas robustas. 0=magnitudes similares; índice geométrico interno."},
            {"key":"pelvis_cycle_consistency_index","label":"Consistencia cíclica de pelvis","value":pelvis_cycle_cons,"unit":"/100","quality":q if np.isfinite(pelvis_cycle_cons) else "No calculable","notes":"Versión 30: similitud de la forma de la oblicuidad pélvica entre zancadas normalizadas. 100=máxima repetibilidad; experimental."},
            {"key":"trunk_lateral_lean_p95_abs","label":"Inclinación lateral absoluta robusta del tronco","value":trunk_abs95,"unit":"°","quality":q,"notes":"Versión 30: P95 (percentil 95) absoluto del eje tronco respecto a la vertical; detecta inclinación sostenida además del ROM."},
            {"key":"trunk_lateral_lean_side_asymmetry_pct","label":"Asimetría de excursión lateral del tronco","value":trunk_side_asym,"unit":"%","quality":q,"notes":"Versión 30: balance entre excursión lateral a ambos lados; descriptor interno 2D."},
            {"key":"trunk_cycle_consistency_index","label":"Consistencia cíclica del tronco","value":trunk_cycle_cons,"unit":"/100","quality":q if np.isfinite(trunk_cycle_cons) else "No calculable","notes":"Versión 30: repetibilidad de la curva de inclinación lateral del tronco entre zancadas."},
            {"key":"shoulder_obliquity_p95_abs","label":"Oblicuidad absoluta robusta de hombros","value":shoulder_abs95,"unit":"°","quality":"Condicionada por ayuda técnica" if assisted else q,"notes":"Versión 30: P95 (percentil 95) absoluto de la línea biacromial; ayuda a detectar sesgos posturales sostenidos."+aid_note},
            {"key":"shoulder_obliquity_side_asymmetry_pct","label":"Asimetría de excursión de hombros","value":shoulder_side_asym,"unit":"%","quality":"Condicionada por ayuda técnica" if assisted else q,"notes":"Versión 30: balance de excursión biacromial entre ambos lados."+aid_note},
            {"key":"shoulder_cycle_consistency_index","label":"Consistencia cíclica de hombros","value":shoulder_cycle_cons,"unit":"/100","quality":"Condicionada por ayuda técnica" if assisted and np.isfinite(shoulder_cycle_cons) else (q if np.isfinite(shoulder_cycle_cons) else "No calculable"),"notes":"Versión 30: repetibilidad de la oblicuidad de hombros entre zancadas."+aid_note},
            {"key":"shoulder_pelvis_rel_p95_abs","label":"Desalineación hombros-pelvis absoluta robusta","value":shoulder_pelvis_abs95,"unit":"°","quality":"Condicionada por ayuda técnica" if assisted else q,"notes":"Versión 30: P95 (percentil 95) absoluto de la diferencia entre oblicuidad de hombros y pelvis; proxy 2D de compensación intersegmentaria."+aid_note},
            {"key":"shoulder_pelvis_cycle_consistency_index","label":"Consistencia cíclica hombros-pelvis","value":shoulder_pelvis_cycle_cons,"unit":"/100","quality":"Condicionada por ayuda técnica" if assisted and np.isfinite(shoulder_pelvis_cycle_cons) else (q if np.isfinite(shoulder_pelvis_cycle_cons) else "No calculable"),"notes":"Versión 30: repetibilidad de la relación hombros-pelvis entre zancadas."+aid_note},
            {"key":"com_lateral_side_asymmetry_pct","label":"Asimetría lateral del COM (Centro de Masa estimado 2D)","value":com_side_asym,"unit":"%","quality":q if np.isfinite(com_side_asym) else "No calculable","notes":"Versión 30: diferencia relativa entre excursión lateral a ambos lados del COM proxy normalizado."},
            {"key":"com_lateral_cycle_consistency_index","label":"Consistencia cíclica del COM (Centro de Masa estimado 2D)","value":com_cycle_cons,"unit":"/100","quality":q if np.isfinite(com_cycle_cons) else "No calculable","notes":"Versión 30: repetibilidad de la trayectoria mediolateral del COM proxy entre zancadas."},
            {"key":"com_bos_cycle_consistency_index","label":"Consistencia cíclica COM (Centro de Masa estimado 2D) / BOS (Base de Sustentación)","value":com_bos_cycle_cons,"unit":"/100","quality":q if np.isfinite(com_bos_cycle_cons) else "No calculable","notes":"Versión 30: repetibilidad de la relación lateral COM/base de sustentación entre zancadas."},
            {"key":"trunk_pelvis_coupling_r","label":"Acoplamiento intersegmentario tronco-pelvis","value":coupling_r,"unit":"r","quality":q,"notes":"Correlación entre inclinación lateral del tronco y oblicuidad pélvica; descriptor 2D de estrategia de compensación."},
            {"key":"trunk_pelvis_phase_deg","label":"Desfase tronco-pelvis estimado","value":phase_deg,"unit":"° ciclo","quality":"Experimental" if np.isfinite(phase_deg) else "No calculable","notes":"v0.9.2: desfase de correlación cruzada normalizado al ciclo IC→IC y expresado entre −180° y +180°; útil para seguimiento, no diagnóstico aislado."},
        ]

    diff_plot = np.asarray(diff, dtype=float).copy()
    if len(diff_plot) == len(straight_mask):
        diff_plot[~np.asarray(straight_mask, bool)] = np.nan
    chart_data = {
        "frame": seg.frame.to_numpy(), "time_s": seg.frame.to_numpy()/fps,
        "straight_walking_valid": np.asarray(straight_mask, bool),
        "straight_block_id": _straight_block_ids(straight_mask),
        "Alternancia D-I": diff_plot,
    }
    if is_frontal:
        chart_data.update({
            "Rodilla frontal izquierda": seg.L_frontal_knee_dev,
            "Rodilla frontal derecha": seg.R_frontal_knee_dev,
            "Orientación pie izquierda": seg.L_foot_progress_proj,
            "Orientación pie derecha": seg.R_foot_progress_proj,
            "Retropié izquierda": seg.L_rearfoot_tilt_proj,
            "Retropié derecha": seg.R_rearfoot_tilt_proj,
            "Oblicuidad pélvica": seg.pelvis_obliquity,
            "Oblicuidad de hombros": seg.shoulder_obliquity,
            "Inclinación lateral del tronco": seg.trunk_lateral_lean,
            "Relación hombros-pelvis": seg.shoulder_pelvis_rel,
            "CoM proxy lateral (px)": seg.com_proxy_centered_px,
            "COM normalizado (% pelvis)": seg.com_proxy_norm_pct_pelvis,
            "Velocidad COM ML (px/s)": seg.com_velocity_px_s,
            "COM respecto BOS (ratio)": seg.com_bos_ratio,
            "Valgo dinámico proyectado I": seg.L_dynamic_valgus,
            "Valgo dinámico proyectado D": seg.R_dynamic_valgus,
        })
    else:
        chart_data.update({
            "Rodilla izquierda": seg.L_knee_flex, "Rodilla derecha": seg.R_knee_flex,
            "Cadera izquierda": seg.L_hip_flex, "Cadera derecha": seg.R_hip_flex,
            "Tobillo izquierda": seg.L_ankle_angle, "Tobillo derecha": seg.R_ankle_angle,
            "Hombro izquierda": seg.L_shoulder_elev, "Hombro derecha": seg.R_shoulder_elev,
        })
    return metrics, pd.DataFrame(chart_data), seg

def metric_value(metrics, key):
    for m in metrics:
        if m["key"] == key:
            return m.get("value")
    return None


def metric_quality(metrics, key):
    for m in metrics:
        if m["key"] == key:
            return m.get("quality")
    return None


def fmt(v, n=1):
    return "—" if v is None or not np.isfinite(v) else f"{v:.{n}f}"


def biomech_summary(metrics, view="", prefix=""):
    """Resumen descriptivo 2D; evita convertir proxies en diagnósticos."""
    def mv(key):
        return metric_value(metrics, f"{prefix}{key}")
    parts = []
    n = mv("step_events_detected"); dur = mv("segment_duration_s"); cad = mv("cadence_exp")
    expected = mv("expected_steps_from_cadence"); err = mv("step_count_consistency_error_pct")
    if n is not None and dur is not None:
        txt = f"En el segmento de {fmt(dur,2)} s se detectaron {fmt(n,0)} eventos de paso/alternancia."
        if cad is not None and np.isfinite(cad):
            txt += f" La cadencia estimada fue {fmt(cad,1)} pasos/min"
            if expected is not None and np.isfinite(expected):
                txt += f", equivalente a {fmt(expected,1)} eventos esperados en esa duración"
            if err is not None and np.isfinite(err):
                txt += f" (discrepancia interna {fmt(err,1)} %)."
            else:
                txt += "."
        parts.append(txt)
    if "Frontal" in (view or ""):
        pkL, pkR = mv("frontal_knee_dev_l_p95"), mv("frontal_knee_dev_r_p95")
        pel = mv("pelvis_obliquity_rom"); sho = mv("shoulder_obliquity_rom"); tr = mv("trunk_lateral_lean_rom")
        if pkL is not None and pkR is not None:
            parts.append(f"El eje frontal de rodilla mostró P95 (percentil 95) de {fmt(pkL,1)}° a izquierda y {fmt(pkR,1)}° a derecha. Son magnitudes proyectadas 2D, no una medición anatómica 3D de valgo/varo.")
        if pel is not None or sho is not None or tr is not None:
            parts.append(f"En el control axial/frontal, el ROM proyectado fue pelvis {fmt(pel,1)}°, hombros {fmt(sho,1)}° y tronco {fmt(tr,1)}°. Estas medidas describen oscilación y alineación en la imagen.")
        rfL, rfR = mv("rearfoot_tilt_l_p95"), mv("rearfoot_tilt_r_p95")
        if rfL is not None and rfR is not None:
            parts.append(f"La inclinación proyectada del retropié alcanzó P95 (percentil 95) absoluto de {fmt(rfL,1)}° a izquierda y {fmt(rfR,1)}° a derecha. Puede informar sobre el patrón de inversión/eversión visible, pero no equivale a pronación 3D.")
    else:
        vals = []
        for k,label in [("hip_flex_diff_p95","cadera"),("knee_flex_diff_p95","rodilla"),("ankle_angle_diff_p95","tobillo"),("shoulder_elev_diff_p95","hombro")]:
            v=mv(k)
            if v is not None and np.isfinite(v): vals.append(f"{label} {fmt(v,1)}°")
        if vals:
            parts.append("Las diferencias D/I de P95 (percentil 95) en la vista lateral fueron: " + ", ".join(vals) + ". Son diferencias 2D proyectadas y deben contextualizarse con el ciclo de marcha y la calidad del tracking.")
    return parts


def get_point(row, name, min_score=0.25):
    try:
        if row[f"{name}_score"] < min_score:
            return None
        return int(round(row[f"{name}_x"])), int(round(row[f"{name}_y"]))
    except Exception:
        return None


def render_angle_video(video_path, full_df, out_path, view="Lateral", assistive_device="Sin ayuda"):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    raw = out_path.with_name(out_path.stem + "_raw.mp4")
    writer = cv2.VideoWriter(str(raw), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w,h))
    enriched = add_angle_columns(full_df)
    if "Frontal" in (view or ""):
        enriched = add_frontal_columns(enriched)
    indexed = enriched.set_index("frame")
    frame_no = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if frame_no in indexed.index:
            row = indexed.loc[frame_no]
            if isinstance(row, pd.DataFrame): row = row.iloc[0]
            for a,b in SKELETON:
                pa, pb = get_point(row,a), get_point(row,b)
                if pa and pb: cv2.line(frame, pa, pb, (40,220,80), 2, cv2.LINE_AA)
            for name in HALPE26:
                p = get_point(row,name)
                if p: cv2.circle(frame,p,3,(0,190,255),-1,cv2.LINE_AA)
            if "Frontal" in (view or ""):
                lines = [
                    f"Rod frontal I/D {row['L_frontal_knee_dev']:.0f}/{row['R_frontal_knee_dev']:.0f} deg",
                    f"Pie orient. I/D {row['L_foot_progress_proj']:.0f}/{row['R_foot_progress_proj']:.0f} deg",
                    f"Retropie I/D {row['L_rearfoot_tilt_proj']:.0f}/{row['R_rearfoot_tilt_proj']:.0f} deg",
                    f"Pelvis {row['pelvis_obliquity']:.0f} deg",
                    f"Hombros {row['shoulder_obliquity']:.0f} deg",
                    f"Tronco {row['trunk_lateral_lean']:.0f} deg",
                    "2D proyectado · no rotacion/pronacion 3D",
                ]
            else:
                lines = [
                    f"Cad I/D {row['L_hip_flex']:.0f}/{row['R_hip_flex']:.0f} deg",
                    f"Rod I/D {row['L_knee_flex']:.0f}/{row['R_knee_flex']:.0f} deg",
                    f"Tob I/D {row['L_ankle_angle']:.0f}/{row['R_ankle_angle']:.0f} deg",
                    f"Hom I/D {row['L_shoulder_elev']:.0f}/{row['R_shoulder_elev']:.0f} deg",
                    "2D proyectado",
                ]
            if assistive_device != "Sin ayuda":
                lines.append(f"Ayuda: {assistive_device}")
            y = 32
            for txt in lines:
                cv2.putText(frame, txt, (12,y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,255), 2, cv2.LINE_AA)
                y += 25
        writer.write(frame)
        frame_no += 1
    cap.release(); writer.release()
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    cmd = f'"{ffmpeg}" -y -loglevel error -i "{raw}" -c:v libx264 -pix_fmt yuv420p -movflags +faststart -an "{out_path}"'
    rc = os.system(cmd)
    try: raw.unlink(missing_ok=True)
    except Exception: pass
    return out_path if rc == 0 and out_path.exists() else None


def cleanup_temp_session(session_dir):
    try:
        shutil.rmtree(session_dir, ignore_errors=True)
        return True
    except Exception:
        return False

def _metric_sentence(metrics, key, title=None, digits=1):
    m=next((x for x in metrics if x.get("key")==key),None)
    if not m or m.get("value") is None:
        return None
    try:
        v=float(m["value"])
        if not np.isfinite(v): return None
    except Exception:
        return None
    label=title or m.get("label",key); unit=m.get("unit","")
    return f"{label}: {v:.{digits}f} {unit}".strip()+f" ({reference_text_for_metric(key)})"

def _fmt_metric_report(metrics, key, label=None, digits=1):
    v=metric_value(metrics,key)
    if v is None or not np.isfinite(v):
        return None
    m=next((x for x in metrics if x.get("key")==key),{})
    unit=m.get("unit","")
    return f"{label or m.get('label',key)}: {float(v):.{digits}f} {unit}".strip()+f" ({reference_text_for_metric(key)})"


def _phase_metric(metrics, base, two_cam=False):
    key=("lateral_"+base) if two_cam and metric_value(metrics,"lateral_"+base) is not None else base
    return key, metric_value(metrics,key)


def generate_reports(metrics, view, patient_code, record_name, assistive_device,
                     patient_age=0, patient_sex="No especificado", record_date=None):
    """Informe clínico estructurado v0.9.3 con metadatos editables, cierre temporal, CV robusto y trazabilidad multipersona."""
    two_cam = any(str(m.get("key", "")).startswith("lateral_") for m in metrics)
    tp = "lateral_" if two_cam else ""
    frontal_prefix = "front_" if two_cam else ""

    def val(key):
        return metric_value(metrics, key)

    def choose(base, prefer_frontal=False):
        candidates = []
        if prefer_frontal:
            candidates.append(frontal_prefix + base)
        candidates += [tp + base, base]
        for k in candidates:
            v = val(k)
            if v is not None and np.isfinite(v):
                return k, float(v)
        return candidates[-1], np.nan

    def ref_for(key):
        # Referencias redactadas para el informe; evita convertir medias/IC en umbrales diagnósticos.
        bare = key.replace("front_", "").replace("lateral_", "")
        if bare == "cadence_exp":
            return "Ref. poblacional contextual: 114.95–118.35 pasos/min (IC95% de la media de cadencia habitual al aire libre en adultos aparentemente sanos; Murtagh et al., Sports Med 2021, PMCID PMC7806575). No es un umbral diagnóstico individual."
        if bare == "regularity_cv":
            return "Sin umbral diagnóstico directamente transferible: en v0.9.2 este CV global es un promedio ponderado de los CV izquierdo y derecho calculados por separado sobre ciclos IC→IC rectilíneos validados y filtrados robustamente; no debe compararse directamente con umbrales publicados para CV de tiempo de paso."
        if bare == "temporal_asymmetry_exp":
            return "Sin umbral universal directamente transferible a este detector 2D. En laboratorio la simetría temporal sana suele aproximarse a 1:1; una cohorte de mujeres jóvenes activas mostró baja asimetría a velocidad preferida (PMCID PMC6335661)."
        if bare in {"stance_time_l_2d", "stance_time_r_2d", "stance_asymmetry_2d"}:
            return "Ref. contextual: el apoyo sano es aproximadamente simétrico entre lados; esta es una estimación cinemática 2D experimental, no una medida de plataforma de fuerzas."
        if bare in {"step_events_detected", "segment_duration_s"}:
            return "Referencia metodológica: control de consistencia interna del propio registro; no existe rango normativo clínico aplicable."
        if bare in {"trendelenburg_drop_l_deg", "trendelenburg_drop_r_deg"}:
            return "Sin umbral diagnóstico 2D universal validado para HALPE26. Descriptor proyectado de caída pélvica durante apoyo monopodal; confirmar clínicamente."
        if bare in {"dynamic_knee_valgus_l_deg", "dynamic_knee_valgus_r_deg"}:
            return "Sin rango normativo validado específicamente para HALPE26 2D. Descriptor de desviación medial proyectada; el valgo dinámico real es multiplanar."
        if bare == "trunk_pelvis_coupling_r":
            return "Sin banda normativa universal para este coeficiente 2D. r≈+1 indica acoplamiento lineal en fase; r≈−1, contrafase; el significado depende de la tarea y del contexto clínico."
        if bare == "trunk_pelvis_phase_deg":
            return "Sin banda normativa universal. En v0.9.2 el desfase se expresa de forma circular entre −180° y +180°; interpretar longitudinalmente y junto con la calidad del tracking."
        return "Sin umbral normativo 2D validado directamente transferible a esta métrica; descriptor proyectado que requiere contextualización clínica."

    # -------- datos principales --------
    kcad, cad = choose("cadence_exp")
    kcv, cv = choose("regularity_cv")
    kasym, asym = choose("temporal_asymmetry_exp")
    kn, n_events = choose("step_events_detected")
    kdur, duration = choose("segment_duration_s")
    ksl, stance_l = choose("stance_time_l_2d")
    ksr, stance_r = choose("stance_time_r_2d")
    ksa, stance_asym = choose("stance_asymmetry_2d")
    _, support_score_l = choose("support_segmentation_score_l")
    _, support_score_r = choose("support_segmentation_score_r")
    support_ok = np.isfinite(stance_l) and np.isfinite(stance_r)

    lines = []
    lines.append("INFORME DE ANÁLISIS BIOMECÁNICO DE LA MARCHA (2D)")
    lines.append("")
    lines.append("Ficha del registro:")
    lines.append(f"• Paciente / Código: {patient_code}")
    if patient_age is not None and int(patient_age or 0) > 0:
        lines.append(f"• Edad: {int(patient_age)} años")
    if patient_sex and str(patient_sex) != "No especificado":
        lines.append(f"• Sexo: {patient_sex}")
    if record_date:
        try:
            date_txt = record_date.strftime("%d/%m/%Y")
        except Exception:
            date_txt = str(record_date)
        lines.append(f"• Fecha del registro: {date_txt}")
    if record_name:
        lines.append(f"• Registro: {record_name}")
    lines.append(f"• Prueba: Análisis de marcha en vista {view}")
    lines.append(f"• Ayuda técnica: {assistive_device}")
    if np.isfinite(n_events) and np.isfinite(duration):
        lines.append(f"• Consistencia interna: {n_events:.0f} eventos detectados en {duration:.2f} s de registro analizado ({ref_for(kn)})")
    else:
        lines.append("• Consistencia interna: no calculable con los datos disponibles (Referencia metodológica: control interno del registro; no existe rango normativo clínico aplicable).")
    if np.isfinite(support_score_l) and np.isfinite(support_score_r):
        qtmp = "suficiente para estimación experimental" if support_ok else "insuficiente: stance/swing anulados"
        lines.append(f"• Control de segmentación apoyo/oscilación: izquierda {support_score_l:.0f}/100; derecha {support_score_r:.0f}/100 — {qtmp} (control matemático interno, no escala clínica).")
    coh_flag = val("temporal_coherence_flag")
    coh_err = val("temporal_coherence_max_error_pct")
    if coh_flag is not None and np.isfinite(coh_flag):
        if coh_flag >= 0.5:
            err_txt = f"{coh_err:.1f}%" if coh_err is not None and np.isfinite(coh_err) else "dentro del límite"
            lines.append(f"• Coherencia temporal v0.9.2: SUPERADA — discordancia máxima {err_txt}. Cadencia, asimetría y doble apoyo pueden publicarse si superan además sus controles específicos.")
        else:
            err_txt = f"{coh_err:.1f}%" if coh_err is not None and np.isfinite(coh_err) else "no cuantificable"
            lines.append(f"• Coherencia temporal v0.9.2: NO SUPERADA — discordancia máxima {err_txt}. El doble apoyo incompatible se suprime. Cadencia, CV y asimetría temporal pueden mantenerse mediante el detector cinemático distal cuando existe periodicidad articular suficiente, siempre con su calidad explícita.")

    # v0.9.0 · trazabilidad del sujeto cuando existe acompañante/terapeuta.
    manual_keys = ["subject_manual_selection_flag","front_subject_manual_selection_flag","lateral_subject_manual_selection_flag"]
    manual_selected = any((val(k) is not None and np.isfinite(val(k)) and val(k) >= 0.5) for k in manual_keys)
    if manual_selected:
        lines.append("• Selección de sujeto: paciente seleccionado manualmente y seguimiento de identidad bloqueado; las demás personas detectadas quedan excluidas del análisis.")
        cont_candidates = [val("identity_continuity_pct"), val("front_identity_continuity_pct"), val("lateral_identity_continuity_pct")]
        cont_candidates = [float(x) for x in cont_candidates if x is not None and np.isfinite(x)]
        amb_candidates = [val("identity_ambiguous_excluded_pct"), val("front_identity_ambiguous_excluded_pct"), val("lateral_identity_ambiguous_excluded_pct")]
        amb_candidates = [float(x) for x in amb_candidates if x is not None and np.isfinite(x)]
        if cont_candidates:
            lines.append(f"• Continuidad de identidad del sujeto: {min(cont_candidates):.1f}% como valor conservador entre vistas.")
        if amb_candidates:
            lines.append(f"• Frames ambiguos excluidos por riesgo de cambio de identidad: {max(amb_candidates):.1f}% como valor conservador entre vistas.")

    lines.append("")
    lines.append("1. PARÁMETROS ESPACIOTEMPORALES (RITMO, VARIABILIDAD Y CARGA)")
    if np.isfinite(cad):
        lines.append(f"• Cadencia estimada: {cad:.1f} pasos/min ({ref_for(kcad)})")
    if np.isfinite(cv):
        lines.append(f"• Variabilidad temporal (CV): {cv:.1f}% ({ref_for(kcv)})")
    if np.isfinite(asym):
        lines.append(f"• Asimetría temporal global: {asym:.1f}% ({ref_for(kasym)})")

    if np.isfinite(stance_l) and np.isfinite(stance_r):
        if stance_l > stance_r:
            dominant = "IZQUIERDA"
        elif stance_r > stance_l:
            dominant = "DERECHA"
        else:
            dominant = "SIMILAR EN AMBAS EXTREMIDADES"
        sa_txt = f"{stance_asym:.1f}%" if np.isfinite(stance_asym) else "no calculable"
        lines.append("• Tiempo de apoyo 2D estimado (fase de carga):")
        lines.append(f"  - Izquierda: {stance_l:.2f} s ({ref_for(ksl)})")
        lines.append(f"  - Derecha: {stance_r:.2f} s ({ref_for(ksr)})")
        lines.append(f"  - Asimetría de carga: {sa_txt}, con mayor tiempo de apoyo estimado en la extremidad {dominant} ({stance_l:.2f} s vs. {stance_r:.2f} s) ({ref_for(ksa)})")
    else:
        lines.append("• Tiempo de apoyo 2D estimado: NO INFORMADO. La segmentación apoyo/oscilación no supera el control de consistencia temporal en ambos lados; se evita publicar una duración potencialmente falsa (estimación markerless 2D, no plataforma de fuerzas).")

    # Opcional doble apoyo si está disponible, siempre con referencia.
    kds, ds = choose("double_support_pct_2d")
    if np.isfinite(ds):
        lines.append(f"• Apoyo bipodal estimado: {ds:.1f}% del segmento/ciclo analizado (Ref. contextual: en marcha adulta habitual el doble apoyo total suele ocupar aproximadamente 20–24% del ciclo; la estimación v0.9.2 usa el mismo dominio de ciclos rectilíneos que cadencia/apoyo y se anula si falla el control físico interno).")

    lines.append("")
    lines.append("2. CINEMÁTICA PROYECTADA 2D (ESTABILIDAD FRONTAL Y APOYO MONOPODAL)")
    frontal_available = ("Frontal" in (view or "")) or any(str(m.get("key", "")).startswith("front_") for m in metrics)
    if frontal_available:
        kpl, drop_l = choose("trendelenburg_drop_l_deg", prefer_frontal=True)
        kpr, drop_r = choose("trendelenburg_drop_r_deg", prefer_frontal=True)
        kvl, valg_l = choose("dynamic_knee_valgus_l_deg", prefer_frontal=True)
        kvr, valg_r = choose("dynamic_knee_valgus_r_deg", prefer_frontal=True)
        lines.append("• Caída pélvica en suspensión (drop pélvico durante apoyo monopodal):")
        lines.append(f"  - Lado Izquierdo: {drop_l:.1f}° ({ref_for(kpl)})" if np.isfinite(drop_l) else "  - Lado Izquierdo: no calculable (Sin umbral diagnóstico 2D universal; confirmar clínicamente).")
        lines.append(f"  - Lado Derecho: {drop_r:.1f}° ({ref_for(kpr)})" if np.isfinite(drop_r) else "  - Lado Derecho: no calculable (Sin umbral diagnóstico 2D universal; confirmar clínicamente).")
        lines.append("• Valgo dinámico proyectado (desviación medial durante carga):")
        lines.append(f"  - Lado Izquierdo: {valg_l:.1f}° ({ref_for(kvl)})" if np.isfinite(valg_l) else "  - Lado Izquierdo: no calculable (Sin rango normativo validado específicamente para HALPE26 2D).")
        lines.append(f"  - Lado Derecho: {valg_r:.1f}° ({ref_for(kvr)})" if np.isfinite(valg_r) else "  - Lado Derecho: no calculable (Sin rango normativo validado específicamente para HALPE26 2D).")
        lines.append("• Precaución metodológica: el valgo/varo representa únicamente la desviación medial/lateral proyectada en el plano frontal y no una medición 3D multiplanar aislada. El drop pélvico es un descriptor 2D durante apoyo monopodal y debe confirmarse mediante exploración clínica cuando tenga relevancia terapéutica.")
    else:
        lines.append("• La vista actual no permite una estimación frontal fiable de drop pélvico o valgo/varo proyectado (Sin umbral normativo 2D aplicable porque la geometría de captura no es comparable).")

    lines.append("")
    lines.append("3. COORDINACIÓN INTERSEGMENTARIA (TRONCO-PELVIS)")
    if frontal_available:
        kcr, coupling = choose("trunk_pelvis_coupling_r", prefer_frontal=True)
        kph, phase = choose("trunk_pelvis_phase_deg", prefer_frontal=True)
        if np.isfinite(coupling):
            lines.append(f"• Acoplamiento tronco-pelvis (r): {coupling:.2f} r ({ref_for(kcr)})")
        else:
            lines.append("• Acoplamiento tronco-pelvis (r): no calculable (Sin banda normativa universal para este descriptor 2D).")
        if np.isfinite(phase):
            lines.append(f"• Desfase tronco-pelvis: {phase:.1f}° de ciclo ({ref_for(kph)})")
        else:
            lines.append("• Desfase tronco-pelvis: no calculable (Sin banda normativa universal; 360° = 1 ciclo de marcha).")
        lines.append("• Interpretación: r próximo a +1 indica que tronco y pelvis oscilan de forma linealmente similar/en fase; r próximo a −1 sugiere contrafase. El desfase cuantifica el retraso/adelanto relativo en grados de ciclo (360° = 1 ciclo completo) y puede ayudar a describir estrategias compensatorias, pero no diagnostica por sí solo un patrón tipo Duchenne.")
    else:
        lines.append("• La coordinación frontal tronco-pelvis no se estima de forma fiable desde una única vista lateral (Sin referencia normativa aplicable).")

    lines.append("")
    lines.append("4. IMPRESIÓN BIOMECÁNICA GENERAL")
    synthesis = []
    if np.isfinite(cad):
        synthesis.append(f"El registro presenta una cadencia estimada de {cad:.1f} pasos/min, que debe interpretarse frente a la referencia poblacional sana solo como contexto y no como objetivo terapéutico automático")
    if np.isfinite(stance_l) and np.isfinite(stance_r):
        dom = "izquierda" if stance_l > stance_r else "derecha" if stance_r > stance_l else "sin predominio lateral claro"
        synthesis.append(f"la estimación temporal por ciclos continuos muestra mayor permanencia en apoyo en la extremidad {dom} ({stance_l:.2f} s izquierda vs. {stance_r:.2f} s derecha)")
    else:
        synthesis.append("el tiempo de apoyo y su direccionalidad no se informan porque la segmentación temporal no alcanzó consistencia suficiente")
    if frontal_available:
        bits=[]
        if 'drop_l' in locals() and np.isfinite(drop_l): bits.append(f"drop pélvico izquierdo {drop_l:.1f}°")
        if 'drop_r' in locals() and np.isfinite(drop_r): bits.append(f"derecho {drop_r:.1f}°")
        if 'valg_l' in locals() and np.isfinite(valg_l): bits.append(f"valgo proyectado izquierdo {valg_l:.1f}°")
        if 'valg_r' in locals() and np.isfinite(valg_r): bits.append(f"derecho {valg_r:.1f}°")
        if bits:
            synthesis.append("en el plano frontal se observan " + ", ".join(bits) + ", todos ellos descriptores 2D sin umbral diagnóstico universal")
        if 'coupling' in locals() and np.isfinite(coupling):
            ptxt = f" y desfase {phase:.1f}°" if 'phase' in locals() and np.isfinite(phase) else ""
            synthesis.append(f"la coordinación tronco-pelvis muestra r={coupling:.2f}{ptxt}, útil principalmente para seguimiento longitudinal y contextualización clínica")
    if not synthesis:
        lines.append("No hay suficientes métricas válidas para generar una síntesis biomecánica cuantitativa. Debe revisarse la calidad del tracking y el intervalo seleccionado.")
    else:
        # 3–4 líneas cortas, unificadas y sin diagnóstico automático.
        lines.append(". ".join(synthesis) + ".")
        lines.append("Los hallazgos deben integrarse con velocidad de marcha, ayuda técnica, exploración neurológica/musculoesquelética y evolución intraindividual; las métricas markerless 2D no sustituyen una evaluación 3D o instrumentada cuando la decisión clínica depende de fuerzas, contacto o cinemática multiplanar.")

    # La salida principal solicitada es el informe clínico. Mantener una versión paciente separada
    # para compatibilidad con la interfaz anterior, pero no mezclarla dentro de los 4 bloques obligatorios.
    patient=[]
    patient.append("VERSIÓN SIMPLIFICADA PARA EL PACIENTE")
    if np.isfinite(cad):
        patient.append(f"En este registro se estimaron {cad:.1f} pasos por minuto. Las cifras de personas sanas se muestran solo como referencia general; en rehabilitación importa especialmente cómo cambia tu marcha respecto a tus propios registros.")
    if np.isfinite(stance_l) and np.isfinite(stance_r):
        side = "izquierda" if stance_l > stance_r else "derecha" if stance_r > stance_l else "ambas de forma parecida"
        patient.append(f"El análisis sugiere que la pierna que permanece apoyada durante más tiempo es la {side}. La estimación utiliza ciclos continuos y se interpreta junto con la exploración clínica.")
    else:
        patient.append("En este registro no se ofrece un tiempo de apoyo por pierna porque el vídeo no permitió segmentar apoyo y oscilación con suficiente consistencia. Es preferible no mostrar una cifra dudosa.")
    if frontal_available:
        patient.append("También se observa cómo se mantiene la pelvis cuando una pierna está en el aire, cómo se alinean las rodillas de frente y cómo se coordinan el tronco y la pelvis. Son medidas proyectadas en una imagen 2D, útiles sobre todo para comparar tu evolución en sesiones realizadas de forma similar.")
    patient.append("El resultado no se interpreta de forma aislada: se combina con tus síntomas, capacidad funcional, ayudas utilizadas y evolución durante el tratamiento.")
    return "\n".join(lines), "\n\n".join(patient)


# ------------------------- v27 · perfil biomecánico dependiente de la vista -------------------------
# Principio metodológico:
# - El índice de capacidad locomotora (ICLM) y el perfil biomecánico son entidades separadas.
# - La cámara frontal NO penaliza la magnitud absoluta del "valgo" 2D ni pie/retropié.
#   Esas variables quedan como descriptores por su dependencia de anatomía, rotación y perspectiva.
# - En frontal única solo se puntúa SIMETRÍA D/I de rodilla proyectada y control pélvico bilateral.
# - En frontal+lateral, el compuesto se apoya principalmente en simetría cinemática sagital
#   (cadera/rodilla/tobillo) y control pélvico frontal. Pie/retropié y valgo frontal son descriptivos.
# - Las bandas 0-100 son operativas/internas y sirven para seguimiento; no son umbrales diagnósticos.


def _metric_record(metrics, key):
    for m in (metrics or []):
        if m.get("key") == key:
            return m
    return None


def _metric_float(metrics, key):
    m = _metric_record(metrics, key)
    if not m:
        return np.nan
    try:
        v = float(m.get("value"))
        return v if np.isfinite(v) else np.nan
    except Exception:
        return np.nan


def _metric_is_usable(metrics, key, allow_experimental=True):
    m = _metric_record(metrics, key)
    if not m:
        return False
    try:
        v=float(m.get("value"))
        if not np.isfinite(v): return False
    except Exception:
        return False
    q=str(m.get("quality","") or "").lower()
    blocked=["no fiable","no calculable","requiere escala","suprim","invalid","fallo"]
    if any(x in q for x in blocked): return False
    if (not allow_experimental) and "experimental" in q: return False
    return True


def _metric_value_any(metrics, keys, allow_experimental=True):
    """Devuelve el primer valor utilizable entre claves candidatas y la clave usada."""
    for key in keys:
        if _metric_is_usable(metrics,key,allow_experimental=allow_experimental):
            return _metric_float(metrics,key), key
    return np.nan, None


def _metric_mean_any(metrics, keys, allow_experimental=True):
    vals=[]
    for key in keys:
        if _metric_is_usable(metrics,key,allow_experimental=allow_experimental):
            v=_metric_float(metrics,key)
            if np.isfinite(v): vals.append(v)
    return float(np.mean(vals)) if vals else np.nan


def _score_lower_better(value, good_max, poor_at):
    try: x=float(value)
    except Exception: return np.nan
    if not np.isfinite(x): return np.nan
    if x <= good_max: return 100.0
    if x >= poor_at: return 0.0
    return float(100.0*(poor_at-x)/(poor_at-good_max))


def _score_higher_better(value, poor_at, good_min):
    try: x=float(value)
    except Exception: return np.nan
    if not np.isfinite(x): return np.nan
    if x >= good_min: return 100.0
    if x <= poor_at: return 0.0
    return float(100.0*(x-poor_at)/(good_min-poor_at))


def _score_band(value, low, high, tolerance):
    try: x=float(value)
    except Exception: return np.nan
    if not np.isfinite(x): return np.nan
    if low <= x <= high: return 100.0
    d=(low-x) if x<low else (x-high)
    return float(np.clip(100.0*(1.0-d/tolerance),0.0,100.0))


def _mean_valid(items, weights=None):
    vals=[]; wts=[]
    if weights is None: weights=[1.0]*len(items)
    for v,w in zip(items,weights):
        try: vv=float(v); ww=float(w)
        except Exception: continue
        if np.isfinite(vv) and np.isfinite(ww) and ww>0:
            vals.append(vv); wts.append(ww)
    if not vals: return np.nan
    return float(np.average(vals,weights=wts))


def _index_classification(score):
    if not np.isfinite(score): return "No calculable"
    if score >= 85: return "Patrón global conservado / alteración mínima"
    if score >= 70: return "Alteración leve"
    if score >= 50: return "Alteración moderada"
    return "Alteración marcada"


def _biomech_classification(score):
    """Lectura del perfil biomecánico: no pretende graduar discapacidad."""
    if not np.isfinite(score): return "No calculable"
    if score >= 85: return "Simetría/control altos en dominios puntuados"
    if score >= 70: return "Discrepancia biomecánica leve"
    if score >= 50: return "Discrepancia biomecánica moderada"
    return "Discrepancia biomecánica marcada"


def _domain_classification(score):
    if not np.isfinite(score): return "Descriptivo / no puntuado"
    if score >= 85: return "Conservado / simétrico"
    if score >= 70: return "Discrepancia leve"
    if score >= 50: return "Discrepancia moderada"
    return "Discrepancia prioritaria"


def _infer_biomech_context(metrics, analysis_mode=None, view=None):
    keys={str(m.get("key","")) for m in (metrics or [])}
    mode=str(analysis_mode or "")
    vw=str(view or "")
    two_cam=mode.startswith("2 cámaras") or any(k.startswith("front_") for k in keys) and any(k.startswith("lateral_") for k in keys)
    if two_cam: return "frontal+lateral"
    if "Lateral" in vw or any(k.startswith("knee_flex_") for k in keys): return "lateral"
    return "frontal"


def _pair_score_from_lr(metrics, lkeys, rkeys, good=3.0, poor=20.0):
    lv,_=_metric_value_any(metrics,lkeys)
    rv,_=_metric_value_any(metrics,rkeys)
    if not (np.isfinite(lv) and np.isfinite(rv)):
        return np.nan, np.nan
    diff=abs(lv-rv)
    return _score_lower_better(diff,good,poor),diff


def _sagittal_joint_symmetry(metrics, prefix, joint, good=3.0, poor=20.0):
    """Simetría D/I usando diferencia de pico y diferencia de ROM; no compara con un ángulo 'normal'."""
    lp=f"{prefix}{joint}_l_p95"; rp=f"{prefix}{joint}_r_p95"
    lr=f"{prefix}{joint}_l_rom"; rr=f"{prefix}{joint}_r_rom"
    pscore,pdiff=_pair_score_from_lr(metrics,[lp],[rp],good,poor)
    rscore,rdiff=_pair_score_from_lr(metrics,[lr],[rr],good,poor)
    score=_mean_valid([pscore,rscore],[0.55,0.45])
    detail=[]
    if np.isfinite(pdiff): detail.append(f"diferencia P95 (percentil 95) D/I {pdiff:.1f}°")
    if np.isfinite(rdiff): detail.append(f"diferencia ROM D/I {rdiff:.1f}°")
    return score," · ".join(detail) if detail else "Sin métricas suficientes"


def compute_biomechanical_profile(metrics, analysis_mode=None, view=None):
    """ICBF (Índice de Control Biomecánico Frontal) / ICS (Índice de Control Cinemático Sagital) 0.4 (v29): perfil dependiente del plano y no compensatorio.

    FRONTAL: pelvis, tronco, COM (Centro de Masa estimado 2D) / BOS (Base de Sustentación) y hombros-pelvis. Cada dominio intenta integrar
    magnitud postural, balance lateral y consistencia cíclica. Rodilla frontal, valgo,
    pie y retropié quedan como descriptores y NO penalizan.

    La agregación frontal reduce la compensación entre dominios: combina media ponderada
    y peor dominio, y aplica techos cuando coexisten varios déficits marcados. Los umbrales
    son operacionales/experimentales para seguimiento, no límites diagnósticos universales.
    """
    context=_infer_biomech_context(metrics,analysis_mode,view)
    domains={}; details={}; weights={}; descriptive={}; excluded=[]; subprofiles={}

    def val(keys):
        v,_=_metric_value_any(metrics,keys); return v

    def high_score(v):
        return float(np.clip(v,0,100)) if np.isfinite(v) else np.nan

    def frontal_block(prefix=""):
        d={}; det={}; w={}

        pel_rom=val([f"{prefix}pelvis_obliquity_rom"])
        pel_abs=val([f"{prefix}pelvis_obliquity_p95_abs"])
        pel_asym=val([f"{prefix}pelvis_obliquity_side_asymmetry_pct"])
        pel_cons=val([f"{prefix}pelvis_cycle_consistency_index"])
        pel_sc=_mean_valid([
            _score_lower_better(pel_abs,2.0,10.0) if np.isfinite(pel_abs) else np.nan,
            _score_lower_better(pel_rom,4.0,16.0) if np.isfinite(pel_rom) else np.nan,
            _score_lower_better(pel_asym,10.0,60.0) if np.isfinite(pel_asym) else np.nan,
            high_score(pel_cons),
        ],[0.40,0.20,0.15,0.25])
        d["Pelvis y transferencia lateral"]=pel_sc
        det["Pelvis y transferencia lateral"]=(f"P95 (percentil 95) abs {pel_abs:.1f}°" if np.isfinite(pel_abs) else "P95 (percentil 95) abs —")+(f" · ROM {pel_rom:.1f}°" if np.isfinite(pel_rom) else "")+(f" · asimetría {pel_asym:.0f}%" if np.isfinite(pel_asym) else "")+(f" · consistencia {pel_cons:.0f}/100" if np.isfinite(pel_cons) else "")
        w["Pelvis y transferencia lateral"]=0.30

        tr_rom=val([f"{prefix}trunk_lateral_lean_rom"])
        tr_abs=val([f"{prefix}trunk_lateral_lean_p95_abs"])
        tr_asym=val([f"{prefix}trunk_lateral_lean_side_asymmetry_pct"])
        tr_cons=val([f"{prefix}trunk_cycle_consistency_index"])
        tr_sc=_mean_valid([
            _score_lower_better(tr_abs,2.0,12.0) if np.isfinite(tr_abs) else np.nan,
            _score_lower_better(tr_rom,4.0,16.0) if np.isfinite(tr_rom) else np.nan,
            _score_lower_better(tr_asym,10.0,60.0) if np.isfinite(tr_asym) else np.nan,
            high_score(tr_cons),
        ],[0.45,0.15,0.15,0.25])
        d["Control lateral del tronco"]=tr_sc
        det["Control lateral del tronco"]=(f"P95 (percentil 95) abs {tr_abs:.1f}°" if np.isfinite(tr_abs) else "P95 (percentil 95) abs —")+(f" · ROM {tr_rom:.1f}°" if np.isfinite(tr_rom) else "")+(f" · asimetría {tr_asym:.0f}%" if np.isfinite(tr_asym) else "")+(f" · consistencia {tr_cons:.0f}/100" if np.isfinite(tr_cons) else "")
        w["Control lateral del tronco"]=0.25

        sho_rom=val([f"{prefix}shoulder_obliquity_rom"])
        sho_abs=val([f"{prefix}shoulder_obliquity_p95_abs"])
        sho_asym=val([f"{prefix}shoulder_obliquity_side_asymmetry_pct"])
        sho_cons=val([f"{prefix}shoulder_cycle_consistency_index"])
        spr_rom=val([f"{prefix}shoulder_pelvis_rel_rom"])
        spr_abs=val([f"{prefix}shoulder_pelvis_rel_p95_abs"])
        spr_cons=val([f"{prefix}shoulder_pelvis_cycle_consistency_index"])
        shoulder_cons=_mean_valid([high_score(sho_cons),high_score(spr_cons)],[0.5,0.5])
        sho_sc=_mean_valid([
            _score_lower_better(sho_abs,3.0,12.0) if np.isfinite(sho_abs) else np.nan,
            _score_lower_better(sho_rom,5.0,18.0) if np.isfinite(sho_rom) else np.nan,
            _score_lower_better(spr_abs,4.0,15.0) if np.isfinite(spr_abs) else np.nan,
            _score_lower_better(sho_asym,10.0,60.0) if np.isfinite(sho_asym) else np.nan,
            shoulder_cons,
        ],[0.30,0.15,0.25,0.10,0.20])
        d["Hombros y relación hombros-pelvis"]=sho_sc
        det["Hombros y relación hombros-pelvis"]=(f"P95 (percentil 95) hombros {sho_abs:.1f}°" if np.isfinite(sho_abs) else "P95 (percentil 95) hombros —")+(f" · ROM {sho_rom:.1f}°" if np.isfinite(sho_rom) else "")+(f" · P95 (percentil 95) hombros-pelvis {spr_abs:.1f}°" if np.isfinite(spr_abs) else "")+(f" · ROM rel {spr_rom:.1f}°" if np.isfinite(spr_rom) else "")+(f" · consistencia {shoulder_cons:.0f}/100" if np.isfinite(shoulder_cons) else "")
        w["Hombros y relación hombros-pelvis"]=0.20

        comn=val([f"{prefix}com_lateral_excursion_norm_pct"])
        bos=val([f"{prefix}com_bos_ratio_p95"])
        com_asym=val([f"{prefix}com_lateral_side_asymmetry_pct"])
        com_cons=val([f"{prefix}com_lateral_cycle_consistency_index"])
        bos_cons=val([f"{prefix}com_bos_cycle_consistency_index"])
        com_cons_comb=_mean_valid([high_score(com_cons),high_score(bos_cons)],[0.55,0.45])
        com_sc=_mean_valid([
            _score_lower_better(comn,30.0,100.0) if np.isfinite(comn) else np.nan,
            _score_lower_better(bos,0.50,1.10) if np.isfinite(bos) else np.nan,
            _score_lower_better(com_asym,10.0,60.0) if np.isfinite(com_asym) else np.nan,
            com_cons_comb,
        ],[0.30,0.30,0.15,0.25])
        d["COM (Centro de Masa estimado 2D) y estabilidad mediolateral"]=com_sc
        det["COM (Centro de Masa estimado 2D) y estabilidad mediolateral"]=(f"excursión {comn:.1f}% pelvis" if np.isfinite(comn) else "excursión —")+(f" · COM (Centro de Masa estimado 2D) / BOS (Base de Sustentación) P95 (percentil 95) {bos:.2f}" if np.isfinite(bos) else "")+(f" · asimetría {com_asym:.0f}%" if np.isfinite(com_asym) else "")+(f" · consistencia {com_cons_comb:.0f}/100" if np.isfinite(com_cons_comb) else "")
        w["COM (Centro de Masa estimado 2D) y estabilidad mediolateral"]=0.25
        return d,det,w

    def score_block(d,w,noncompensatory=False):
        av={k:float(v) for k,v in d.items() if np.isfinite(v)}
        wa=sum(w.get(k,0) for k in av); wt=sum(w.values())
        weighted=float(sum(av[k]*w[k] for k in av)/wa) if wa>0 else np.nan
        cov=100*wa/wt if wt>0 else 0.0
        if cov<45: return np.nan,cov,[]
        caps=[]
        if noncompensatory and av:
            weakest=min(av.values())
            score=0.65*weighted+0.35*weakest
            nlow=sum(v<50 for v in av.values())
            if nlow>=3 and score>39: score=39.0; caps.append("≥3 dominios frontales <50/100")
            elif nlow>=2 and score>49: score=49.0; caps.append("≥2 dominios frontales <50/100")
            elif weakest<35 and score>59: score=59.0; caps.append("un dominio frontal <35/100")
            return float(score),cov,caps
        return weighted,cov,caps

    frontal_caps=[]
    if context == "frontal":
        domains,details,weights=frontal_block("")
        score,coverage,frontal_caps=score_block(domains,weights,True)
        subprofiles["Control biomecánico frontal"]={"score":score,"coverage_pct":coverage,"domains":domains,"weights":weights,"details":details}
        descriptive["Rodilla frontal proyectada"]="Descriptor clínico D/I. Valgo/varo absoluto no penaliza por morfología, sexo, rotación y perspectiva 2D."
        descriptive["Pie y retropié"]="Descriptivos; excluidos del ICBF frontal por perspectiva y validez limitada del proxy distal."
        descriptive["Coordinación tronco-pelvis"]="r y fase se muestran como estrategia de coordinación, pero no puntúan por ausencia de dirección normativa universal."
        excluded += ["valgo/varo frontal absoluto","pie/retropié","acoplamiento tronco-pelvis como puntuación"]
        profile_name="ICBF (Índice de Control Biomecánico Frontal) · Control biomecánico frontal"
        method_note=("Plano frontal/posterior · Versión 30: pelvis (30%), tronco (25%), COM (Centro de Masa estimado 2D) / BOS (Base de Sustentación) (25%) y hombros-pelvis (20%). "
                     "Cada dominio integra, cuando existe, magnitud postural, balance lateral y consistencia cíclica. "
                     "La agregación es deliberadamente no compensatoria: 65% media ponderada + 35% peor dominio, con techos si coexisten varios déficits. "
                     "Rodilla frontal y pie/retropié no penalizan.")

    elif context == "frontal+lateral":
        fd,fdet,fw=frontal_block("front_")
        fscore,fcov,frontal_caps=score_block(fd,fw,True)
        subprofiles["Control biomecánico frontal"]={"score":fscore,"coverage_pct":fcov,"domains":fd,"weights":fw,"details":fdet}

        sd={}; sdet={}; sw={}
        for name,joint,w in [("Cadera sagital D/I","hip_flex",0.35),("Rodilla sagital D/I","knee_flex",0.35),("Tobillo-pie sagital D/I","ankle_angle",0.30)]:
            sc,det=_sagittal_joint_symmetry(metrics,"lateral_",joint,3.0,20.0)
            sd[name]=sc; sdet[name]=det; sw[name]=w
        sscore,scov,_=score_block(sd,sw,False)
        subprofiles["Control cinemático sagital"]={"score":sscore,"coverage_pct":scov,"domains":sd,"weights":sw,"details":sdet}

        score=_mean_valid([fscore,sscore],[0.60,0.40]); coverage=_mean_valid([fcov,scov],[0.60,0.40])
        domains={**{f"Frontal · {k}":v for k,v in fd.items()},**{f"Sagital · {k}":v for k,v in sd.items()}}
        details={**{f"Frontal · {k}":v for k,v in fdet.items()},**{f"Sagital · {k}":v for k,v in sdet.items()}}
        weights={**{f"Frontal · {k}":0.60*v for k,v in fw.items()},**{f"Sagital · {k}":0.40*v for k,v in sw.items()}}
        descriptive["Rodilla frontal proyectada"]="No penaliza por magnitud absoluta; la cámara lateral aporta la comparación sagital D/I."
        descriptive["Pie/retropié frontal"]="No puntúa. El tobillo-pie lateral entra solo por simetría cinemática sagital D/I."
        descriptive["Coordinación tronco-pelvis"]="Descriptor; no puntuado por falta de umbral normativo universal."
        excluded += ["valgo/varo frontal absoluto","pie/retropié frontal","acoplamiento tronco-pelvis como puntuación"]
        profile_name="Perfil combinado 2D (frontal + sagital)"
        method_note=("Tracking frontal/posterior + lateral · Versión 30: se muestran DOS perfiles independientes. El frontal usa la arquitectura postural no compensatoria (pelvis, tronco, hombros y COM); "
                     "el lateral resume simetría sagital de cadera, rodilla y tobillo-pie. El valor combinado 60/40 se conserva solo para compatibilidad/exportación.")

    else:
        for name,joint,w in [("Cadera sagital D/I","hip_flex",0.35),("Rodilla sagital D/I","knee_flex",0.35),("Tobillo-pie sagital D/I","ankle_angle",0.30)]:
            sc,det=_sagittal_joint_symmetry(metrics,"",joint,3.0,20.0)
            domains[name]=sc; details[name]=det; weights[name]=w
        score,coverage,_=score_block(domains,weights,False)
        subprofiles["Control cinemático sagital"]={"score":score,"coverage_pct":coverage,"domains":domains,"weights":weights,"details":details}
        descriptive["Pelvis/COM frontal"]="No puntuables sin vista frontal/posterior."
        excluded += ["pelvis frontal","COM (Centro de Masa estimado 2D) / BOS (Base de Sustentación) frontal","valgo frontal","retropié frontal"]
        profile_name="ICS (Índice de Control Cinemático Sagital) · Control cinemático sagital"
        method_note="Lateral única: se limita a simetría sagital bilateral de cadera, rodilla y tobillo-pie."

    available={k:float(v) for k,v in domains.items() if np.isfinite(v)}
    tracking=_metric_mean_any(metrics,["tracking_mean","front_tracking_mean","lateral_tracking_mean"])
    straight=_metric_mean_any(metrics,["straight_walking_usable_pct","front_straight_walking_usable_pct","lateral_straight_walking_usable_pct"])
    qparts=[coverage]
    if np.isfinite(tracking): qparts.append(np.clip(tracking*100.0,0,100))
    if np.isfinite(straight): qparts.append(np.clip(straight,0,100))
    confidence=float(np.mean(qparts)) if qparts else np.nan
    conf_label=("Alta" if confidence>=85 else "Moderada" if confidence>=70 else "Limitada") if np.isfinite(confidence) else "No calculable"
    ranked=sorted(available.items(),key=lambda kv:kv[1])
    return {"global_score":score,"classification":_biomech_classification(score),"domains":domains,"domain_weights":weights,
            "details":details,"descriptive":descriptive,"excluded":excluded,"coverage_pct":float(coverage),
            "confidence_score":confidence,"confidence_label":conf_label,"ranked_deficits":ranked,"context":context,
            "profile_name":profile_name,"method_note":method_note,"subprofiles":subprofiles,"frontal_caps":frontal_caps,
            "version":"ICBF (Índice de Control Biomecánico Frontal) / ICS (Índice de Control Cinemático Sagital) 0.4 experimental"}

def compute_global_gait_index(metrics, analysis_mode=None, view=None):
    """Compatibilidad: desde v29 el antiguo IGPM queda sustituido por perfiles dependientes de la vista."""
    return compute_biomechanical_profile(metrics,analysis_mode=analysis_mode,view=view)

def _functional_support_score(status):
    """Puntuación operacional de independencia. Entrada clínica/manual, no inferida del vídeo."""
    return {
        "Independiente": 100.0,
        "Supervisión": 75.0,
        "Contacto ocasional con apoyo externo": 55.0,
        "Apoyo externo frecuente": 30.0,
        "Ayuda física de otra persona": 10.0,
    }.get(str(status), np.nan)


def compute_locomotor_capacity_index(metrics, functional_support="Independiente", measured_gait_speed=0.0):
    """ICLM (Índice de Capacidad Locomotora) 0.1 experimental: capacidad locomotora, separada del perfil biomecánico.

    Prioriza velocidad (si está medida), consistencia/variabilidad, simetría e independencia.
    Un déficit locomotor dominante limita la clasificación; una buena alineación segmentaria
    NO puede compensarlo. Los fallos de detección/QC reducen confianza, no la función.
    """
    parts={}; weights={}; details={}
    cv,_=_metric_value_any(metrics,["regularity_cv","lateral_regularity_cv","front_regularity_cv"])
    sim,_=_metric_value_any(metrics,["step_to_step_similarity_index","lateral_step_to_step_similarity_index","front_step_to_step_similarity_index"])
    if np.isfinite(cv) or np.isfinite(sim):
        parts["Regularidad y consistencia"]=_mean_valid([
            _score_lower_better(cv,5.0,20.0) if np.isfinite(cv) else np.nan,
            np.clip(sim,0,100) if np.isfinite(sim) else np.nan],[0.55,0.45])
        weights["Regularidad y consistencia"]=0.30
        details["Regularidad y consistencia"]=(f"CV {cv:.1f}%" if np.isfinite(cv) else "CV —")+(f" · consistencia {sim:.1f}/100" if np.isfinite(sim) else "")
    ta,_=_metric_value_any(metrics,["temporal_asymmetry_exp","lateral_temporal_asymmetry_exp","front_temporal_asymmetry_exp"])
    sa,_=_metric_value_any(metrics,["stance_asymmetry_2d","lateral_stance_asymmetry_2d","front_stance_asymmetry_2d"])
    if np.isfinite(ta) or np.isfinite(sa):
        parts["Simetría locomotora"]=_mean_valid([
            _score_lower_better(ta,5.0,30.0) if np.isfinite(ta) else np.nan,
            _score_lower_better(sa,5.0,40.0) if np.isfinite(sa) else np.nan],[0.70,0.30])
        weights["Simetría locomotora"]=0.25
        details["Simetría locomotora"]=(f"asimetría temporal {ta:.1f}%" if np.isfinite(ta) else "")+(f" · apoyo {sa:.1f}%" if np.isfinite(sa) else "")
    try: speed=float(measured_gait_speed)
    except Exception: speed=0.0
    if np.isfinite(speed) and speed>0:
        # Banda operacional para seguimiento adulto; no se etiqueta como normalidad diagnóstica.
        parts["Velocidad de marcha"]=_score_higher_better(speed,0.20,1.20)
        weights["Velocidad de marcha"]=0.30
        details["Velocidad de marcha"]=f"{speed:.2f} m/s · valor medido/calibrado"
    ss=_functional_support_score(functional_support)
    if np.isfinite(ss):
        parts["Independencia durante la marcha"]=ss
        weights["Independencia durante la marcha"]=0.15
        details["Independencia durante la marcha"]=str(functional_support)
    available={k:v for k,v in parts.items() if np.isfinite(v)}
    sw=sum(weights[k] for k in available)
    raw=float(sum(available[k]*weights[k] for k in available)/sw) if sw>0 else np.nan
    score=raw
    caps=[]
    # Regla de déficit dominante: evita que dominios conservados oculten incapacidad locomotora.
    if np.isfinite(speed) and speed>0:
        if speed < 0.40: caps.append((49.0,"velocidad <0,40 m/s"))
        elif speed < 0.80: caps.append((69.0,"velocidad <0,80 m/s"))
    if functional_support in ("Apoyo externo frecuente","Ayuda física de otra persona"):
        caps.append((49.0,"dependencia de apoyo externo/ayuda física"))
    elif functional_support == "Contacto ocasional con apoyo externo":
        caps.append((69.0,"contacto ocasional con apoyo externo"))
    if np.isfinite(parts.get("Regularidad y consistencia",np.nan)) and parts["Regularidad y consistencia"]<40:
        caps.append((69.0,"alteración marcada de regularidad/consistencia"))
    if np.isfinite(parts.get("Simetría locomotora",np.nan)) and parts["Simetría locomotora"]<40:
        caps.append((69.0,"alteración marcada de simetría locomotora"))
    if np.isfinite(score) and caps:
        score=min(score,min(c[0] for c in caps))
    # QC: discrepancias de eventos afectan confianza, nunca se interpretan como déficit del paciente.
    count_err=_metric_mean_any(metrics,["step_count_consistency_error_pct","lateral_step_count_consistency_error_pct","front_step_count_consistency_error_pct"])
    tracking=_metric_mean_any(metrics,["tracking_mean","front_tracking_mean","lateral_tracking_mean"])
    straight=_metric_mean_any(metrics,["straight_walking_usable_pct","front_straight_walking_usable_pct","lateral_straight_walking_usable_pct"])
    q=[100.0]
    if np.isfinite(count_err): q.append(np.clip(100-count_err,0,100))
    if np.isfinite(tracking): q.append(np.clip(tracking*100,0,100))
    if np.isfinite(straight): q.append(np.clip(straight,0,100))
    confidence=float(np.mean(q)) if q else np.nan
    conf_label="Alta" if confidence>=85 else "Moderada" if confidence>=70 else "Limitada"
    return {"score":score,"raw_score":raw,"classification":_index_classification(score),"components":available,
            "details":details,"caps":[c[1] for c in caps],"confidence_score":confidence,"confidence_label":conf_label,
            "version":"ICLM (Índice de Capacidad Locomotora) 0.1 experimental"}

def augment_metrics_with_gait_index(metrics, functional_support="Independiente", measured_gait_speed=0.0, analysis_mode=None, view=None):
    """Añade ICLM (Índice de Capacidad Locomotora) + perfil biomecánico IBV (Índice Biomecánico dependiente de la Vista) 0.2 sin duplicar claves derivadas."""
    derived_prefixes=("gait_index_","biomech_profile_","locomotor_")
    derived_exact={"measured_gait_speed_ms","functional_support_score"}
    base=[m for m in (metrics or []) if not str(m.get("key","")).startswith(derived_prefixes) and str(m.get("key","")) not in derived_exact]
    idx=compute_biomechanical_profile(base,analysis_mode=analysis_mode,view=view)
    loc=compute_locomotor_capacity_index(base,functional_support,measured_gait_speed)
    rows=[
        {"key":"locomotor_capacity_index","label":"Índice de Capacidad Locomotora","value":loc["score"],"unit":"/100","quality":loc["confidence_label"],"notes":"ICLM (Índice de Capacidad Locomotora) 0.1 experimental. Separa capacidad funcional del perfil biomecánico; aplica reglas de déficit dominante."},
        {"key":"locomotor_capacity_raw","label":"Capacidad locomotora antes de reglas dominantes","value":loc["raw_score"],"unit":"/100","quality":loc["confidence_label"],"notes":"Valor compuesto previo a límites por velocidad/apoyo/alteración locomotora dominante."},
        {"key":"locomotor_measurement_confidence","label":"Confianza de medición locomotora","value":loc["confidence_score"],"unit":"/100","quality":loc["confidence_label"],"notes":"QC separado de función: los fallos de detección reducen confianza y no empeoran artificialmente la capacidad del paciente."},
        {"key":"measured_gait_speed_ms","label":"Velocidad de marcha medida","value":float(measured_gait_speed) if float(measured_gait_speed or 0)>0 else np.nan,"unit":"m/s","quality":"Medida externa/calibrada" if float(measured_gait_speed or 0)>0 else "No disponible","notes":"Solo se usa si se introduce una medida válida; no se inventa velocidad métrica sin calibración."},
        {"key":"functional_support_score","label":"Independencia funcional durante la marcha","value":_functional_support_score(functional_support),"unit":"/100","quality":"Entrada clínica","notes":str(functional_support)+". Variable clínica/manual; no inferida automáticamente del vídeo."},
        {"key":"biomech_profile_index","label":idx["profile_name"],"value":idx["global_score"],"unit":"/100","quality":idx["confidence_label"],"notes":idx["version"]+". "+idx["method_note"]},
        {"key":"biomech_profile_coverage_pct","label":"Cobertura del perfil biomecánico","value":idx["coverage_pct"],"unit":"% peso disponible","quality":idx["confidence_label"],"notes":"Cobertura de los dominios que son puntuables para la vista utilizada."},
        {"key":"biomech_profile_confidence","label":"Confianza interna del perfil biomecánico","value":idx["confidence_score"],"unit":"/100","quality":idx["confidence_label"],"notes":"Calidad de medición separada del patrón biomecánico."},
        # Compatibilidad de exportación con informes previos: misma clave histórica, nueva semántica documentada.
        {"key":"gait_index_global","label":"Perfil biomecánico dependiente de la vista (compatibilidad)","value":idx["global_score"],"unit":"/100","quality":idx["confidence_label"],"notes":"Desde v27 corresponde a IBV (Índice Biomecánico dependiente de la Vista) 0.2, no al antiguo IGPM (Índice Global y Perfil de Marcha) 0.1."},
        {"key":"gait_index_coverage_pct","label":"Cobertura perfil biomecánico (compatibilidad)","value":idx["coverage_pct"],"unit":"% peso disponible","quality":idx["confidence_label"],"notes":"Alias histórico del perfil biomecánico IBV (Índice Biomecánico dependiente de la Vista) 0.2."},
        {"key":"gait_index_confidence","label":"Confianza perfil biomecánico (compatibilidad)","value":idx["confidence_score"],"unit":"/100","quality":idx["confidence_label"],"notes":"Alias histórico del perfil biomecánico IBV (Índice Biomecánico dependiente de la Vista) 0.2."},
    ]
    for i,(name,score) in enumerate(idx["domains"].items(),1):
        rows.append({"key":f"biomech_profile_domain_{i}","label":f"Perfil biomecánico · {name}","value":score,"unit":"/100","quality":_domain_classification(score),"notes":idx["details"].get(name,"")})
    return base+rows


# ------------------------- UI -------------------------
# ------------------------- UI -------------------------
st.title("PhysioSentinel Gait")
st.caption(f"Versión {APP_VERSION} · V56 CORREGIDA 3 congelada · V57 visor 3D · V58 calibración clínica estéreo ChArUco")

if sb_ready():
    st.success("☁️ Supabase conectado: pacientes, sesiones, métricas y perfiles de calibración pueden persistir. Los vídeos NO se guardan.")
else:
    st.error("Supabase aún no está configurado. Añade SUPABASE_URL y SUPABASE_SERVICE_ROLE_KEY en Streamlit Secrets.")

# Estado base
for k, v in {
    "pose_done": False, "metrics_done": False, "temp_deleted": False,
    "analysis_df": None, "analysis_df2": None,
    "annotated_video_bytes": None, "annotated_video2_bytes": None,
    "sync_offset_auto_s": 0.0, "sync_offset_user_s": 0.0,
    "sync_correlation": np.nan, "sync_quality": "No calculable",
                    "biplanar_window_cam1": None, "biplanar_window_cam2": None,
                    "biplanar_cycle_pairs": [], "biplanar_cycle_quality": "No calculable",
    "camera1_quality": None, "camera2_quality": None,
    "pose_raw_done": False, "subject_locked": False,
    "pose_analysis_running": False, "pose_execution_id": None,
    "subject_selection_cam1": None, "subject_selection_cam2": None,
    "subject_preview_cam1": None, "subject_preview_cam2": None,
    "tracking_info_cam1": None, "tracking_info_cam2": None,
    "pose_json1": None, "pose_json2": None,
    # v0.9.3 · metadatos editables sin reiniciar/reprocesar el análisis
    "patient_code": "Prueba",
    "record_name": "Marcha",
    "patient_age": 0,
    "patient_sex": "No especificado",
    "record_date": datetime.now().date(),
    # v0.10.1 · snapshot ligero para el analizador de ciclo
    "cycle_seg": None,
    "cycle_seg_front": None,
    "cycle_seg_lateral": None,
    "cycle_fps": None,
    "cycle_fps_front": None,
    "cycle_fps_lateral": None,
    "cycle_support_left": None,
    "cycle_support_right": None,
    "cycle_support_left_front": None,
    "cycle_support_right_front": None,
    "cycle_support_left_lateral": None,
    "cycle_support_right_lateral": None,
    "cycle_source_view": None,
    # v0.10.3 · vídeo limpio de sesión para pestaña 9 (no Supabase)
    "cycle_video_clean_bytes": None,
    "cycle_video2_clean_bytes": None,
    "selected_subject_label_cam1": None,
    "selected_subject_label_cam2": None,
    "cycle_phase_review_base": None,
    "cycle_phase_review_bytes": None,
    "cycle_phase_review_speed": None,
}.items():
    if k not in st.session_state:
        st.session_state[k] = v

# Perfiles de calibración disponibles
calibration_names = []
if sb_ready():
    try:
        calibration_names = [x["name"] for x in sb_list_calibrations()]
    except Exception:
        calibration_names = []

with st.sidebar:
    st.header("Sesión")

    # v0.9.3 · Los datos personales se editan dentro de un formulario.
    # Escribir en los campos NO provoca reruns sucesivos. Solo el botón Guardar
    # actualiza los metadatos y nunca invalida vídeo, pose, métricas ni segmento.
    with st.expander("👤 Datos del paciente / registro", expanded=True):
        with st.form("patient_metadata_form", clear_on_submit=False):
            patient_edit = st.text_input(
                "Paciente / código",
                value=str(st.session_state.get("patient_code", "Prueba")),
            )
            record_edit = st.text_input(
                "Nombre del registro",
                value=str(st.session_state.get("record_name", "Marcha")),
            )
            age_edit = st.number_input(
                "Edad",
                min_value=0, max_value=120,
                value=int(st.session_state.get("patient_age", 0) or 0),
                step=1,
                help="0 = no especificada.",
            )
            sex_options = ["No especificado", "Mujer", "Hombre", "Otro / no binario"]
            current_sex = str(st.session_state.get("patient_sex", "No especificado"))
            sex_index = sex_options.index(current_sex) if current_sex in sex_options else 0
            sex_edit = st.selectbox("Sexo", sex_options, index=sex_index)
            date_edit = st.date_input(
                "Fecha del registro",
                value=st.session_state.get("record_date", datetime.now().date()),
            )
            save_metadata = st.form_submit_button(
                "💾 Guardar datos",
                type="primary",
                use_container_width=True,
            )

        if save_metadata:
            old_code = str(st.session_state.get("patient_code", ""))
            st.session_state["patient_code"] = patient_edit.strip() or "Prueba"
            st.session_state["record_name"] = record_edit.strip() or "Marcha"
            st.session_state["patient_age"] = int(age_edit)
            st.session_state["patient_sex"] = sex_edit
            st.session_state["record_date"] = date_edit

            # Regenerar textos del informe con los nuevos metadatos,
            # sin tocar resultados ni archivos de pose.
            st.session_state.pop("report_technical", None)
            st.session_state.pop("report_patient", None)

            # Si la sesión ya existe en Supabase, reasignar código y nombre
            # sin crear una sesión nueva ni reprocesar el vídeo.
            sid = st.session_state.get("cloud_session_id")
            if sid and sb_ready():
                try:
                    sb_update_session_identity(
                        sid,
                        patient_code=st.session_state["patient_code"],
                        record_name=st.session_state["record_name"],
                    )
                except Exception as e:
                    st.warning(f"Los datos se han actualizado en esta sesión, pero Supabase no pudo sincronizar código/nombre del registro: {e}")

            st.success("Datos actualizados. El análisis biomecánico se conserva intacto.")

    patient = str(st.session_state.get("patient_code", "Prueba"))
    record = str(st.session_state.get("record_name", "Marcha"))
    patient_age = int(st.session_state.get("patient_age", 0) or 0)
    patient_sex = str(st.session_state.get("patient_sex", "No especificado"))
    record_date = st.session_state.get("record_date", datetime.now().date())

    st.caption("Usa preferentemente un código seudonimizado en lugar de nombre y apellidos.")
    st.divider()
    analysis_level = st.radio(
        "Nivel de análisis",
        [
            "Nivel 1 · 2D · una cámara",
            "Nivel 2 · Biplanar + 3D estimado · frontal/posterior + lateral",
            "Nivel 3 · 3D calibrado cuantitativo · frontal/posterior + lateral",
        ],
        index=0,
        help=(
            "Nivel 1: análisis 2D de una vista. Nivel 2: integra dos vistas, sincroniza el ciclo y genera "
            "un esqueleto 3D estimado/no métrico sin calibración. Nivel 3: añade triangulación espacial "
            "X/Y/Z cuantitativa y requiere una calibración válida de las cámaras."
        ),
    )
    if analysis_level.startswith("Nivel 1"):
        mode = "1 cámara · 2D"
        st.caption("🟢 2D: no requiere calibración de cámaras.")
    elif analysis_level.startswith("Nivel 2"):
        mode = "2 cámaras · frontal/posterior + lateral · sincronización biplanar"
        st.caption("🟣 Biplanar + 3D estimado: combina ambas vistas y puede recrear un esqueleto 3D cualitativo/no métrico sin calibración.")
    else:
        mode = "2 cámaras · frontal/posterior + lateral · sincronización biplanar"
        st.caption("🔵 3D calibrado cuantitativo: conserva el 2D/biplanar y habilita triangulación X/Y/Z métrica cuando existe calibración válida; si falta, usa 3D estimado como respaldo visual.")
    is_two_cam = not analysis_level.startswith("Nivel 1")
    wants_3d = analysis_level.startswith("Nivel 3")
    st.session_state["analysis_level"] = analysis_level
    st.session_state["wants_3d"] = wants_3d
    view = st.radio("Vista", ["Frontal/posterior", "Lateral"], index=0) if not is_two_cam else "Frontal+Lateral"
    frontal_orientation = st.selectbox(
        "Sentido de la toma frontal/posterior",
        ["No especificada", "Frontal", "Posterior", "Mixta/ida-vuelta"],
        index=0,
    ) if "Frontal" in view else "No aplica"
    assistive_device = st.selectbox("Ayuda técnica", ASSISTIVE_OPTIONS, index=0)
    if assistive_device != "Sin ayuda":
        st.caption("La app medirá la visibilidad por regiones y marcará métricas condicionadas por posibles oclusiones.")
    functional_support = st.selectbox(
        "Independencia funcional durante la marcha", FUNCTIONAL_SUPPORT_OPTIONS, index=0,
        help="Registra apoyo en pared/barandilla o ayuda humana aunque no se considere una ayuda técnica convencional."
    )
    measured_gait_speed = st.number_input(
        "Velocidad de marcha medida (m/s, opcional)", min_value=0.0, value=0.0, step=0.05, format="%.2f",
        help="Introduce una velocidad obtenida con distancia/tiempo o sistema calibrado. 0 = no disponible. No se estima m/s sin calibración válida."
    )
    st.session_state["functional_support"] = functional_support
    st.session_state["measured_gait_speed"] = float(measured_gait_speed)
    subject_mode = st.radio(
        "Sujeto biomecánico",
        ["Automático · una sola persona", "Selección manual · multipersona"],
        index=0,
        help="Usa selección manual cuando el paciente camina acompañado, asistido o estrechamente supervisado."
    )
    manual_subject_mode = subject_mode.startswith("Selección manual")
    if manual_subject_mode:
        st.caption("v0.9.3: primero se detectan las personas; tú eliges al paciente y después la identidad queda bloqueada. Los frames ambiguos se excluyen.")
    if is_two_cam:
        opts = ["Sin calibración"] + calibration_names
        default_cal = st.session_state.get("selected_calibration_name") or "Sin calibración"
        try:
            idx = opts.index(default_cal)
        except ValueError:
            idx = 0
        selected_calibration = st.selectbox("Perfil de calibración 3D", opts, index=idx)
        if wants_3d and selected_calibration == "Sin calibración":
            st.warning("Nivel 3 seleccionado sin calibración: la triangulación X/Y/Z cuantitativa permanecerá desactivada, pero V60 podrá generar el 3D estimado/no métrico a partir de las dos vistas sincronizadas.")
        elif (not wants_3d) and selected_calibration != "Sin calibración":
            st.info("Hay una calibración disponible, pero este nivel no exige reconstrucción 3D.")
        st.session_state.selected_calibration_name = None if selected_calibration == "Sin calibración" else selected_calibration
    else:
        selected_calibration = "Sin calibración"
        st.session_state.selected_calibration_name = None
    st.divider()
    st.markdown("**Escala espacial frontal (opcional)**")
    scale_cm_per_px = st.number_input("cm por píxel", min_value=0.0, value=float(st.session_state.get("scale_cm_per_px",0.0)), step=0.001, format="%.4f", help="Solo si dispones de una calibración espacial válida en el plano de marcha. 0 = sin escala métrica.")
    st.session_state.scale_cm_per_px=float(scale_cm_per_px)
    st.caption("Sin escala válida, CoM y BoS se mantienen como proxies relativos/píxel y no se convierten a cm.")
    st.divider()
    st.markdown("**Motor interno**")
    st.write("Pose2Sim + RTMPose")
    st.write("Body_with_feet / HALPE26")
    st.caption("Vídeos → /tmp → pose 2D → sincronización → resultados → Supabase → eliminación")



def _annex_group(key):
    k=str(key or "")
    if any(x in k for x in ["tracking","visibility","straight_walking","turn_transition","segmentation_score","coherence","crosscheck","unsupported_flight","internal_discrepancy","candidate_","step_count_consistency","support_event"]):
        return "1 · Control de calidad"
    if any(x in k for x in ["cadence","regularity","step_to_step","alternation","stride_cycle","temporal_asymmetry","valid_cycles","cycle_outliers","step_events","segment_duration","valid_straight_duration","expected_steps"]):
        return "2 · Ritmo, regularidad y consistencia"
    if any(x in k for x in ["stance","swing","double_support","initial_contact","loading_","terminal_foot"]):
        return "3 · Apoyo, oscilación e IC/TO"
    if any(x in k for x in ["knee","valgus","foot_progress","rearfoot"]):
        return "4 · Miembro inferior y pie"
    if any(x in k for x in ["pelvis","shoulder","trunk"]):
        return "5 · Pelvis y tronco"
    if any(x in k for x in ["com_","bos_","base_width"]):
        return "6 · COM y estabilidad dinámica"
    if k.startswith("gait_index") or k.startswith("biomech_profile") or k.startswith("locomotor_"):
        return "7 · Índices funcionales y biomecánicos"
    return "8 · Otras variables técnicas"


def _patient_explanation_for_metric(m):
    k=str(m.get("key", "")); label=str(m.get("label", k))
    exact={
        "cadence_exp":"Ritmo de marcha: cuántos pasos realiza aproximadamente en un minuto.",
        "regularity_cv":"Indica cuánto varía la duración de los ciclos. Cuanto menor es, más repetitiva es la marcha.",
        "step_to_step_similarity_index":"Resume de 0 a 100 cuánto se parece cada paso al inmediatamente anterior en su duración.",
        "step_to_step_change_pct":"Mide cuánto cambia, en porcentaje, un intervalo de paso respecto al anterior. Menor suele significar mayor consistencia.",
        "temporal_asymmetry_exp":"Compara el comportamiento temporal de ambas piernas. Un valor bajo indica mayor similitud entre lados.",
        "com_lateral_excursion_norm_pct":"Cuánto se desplaza el centro corporal estimado de lado a lado, ajustado al tamaño aparente de la pelvis.",
        "com_velocity_ml_px_s":"Rapidez con la que el centro corporal estimado se mueve lateralmente en la imagen.",
        "com_bos_ratio_p95":"Indica hasta qué punto el centro corporal estimado se desplaza hacia los límites de la base formada por los pies.",
        "com_bos_ratio_mean_abs":"Indica cuánto se aleja habitualmente el centro corporal estimado del centro de la base de apoyo.",
        "good_frames_pct":"Porcentaje del vídeo en el que las articulaciones principales pudieron seguirse con suficiente confianza.",
        "foot_visibility_pct":"Porcentaje del vídeo en el que pie y tobillo se ven con suficiente calidad para analizarlos.",
        "tracking_mean":"Confianza media con la que el sistema ha localizado las articulaciones en el vídeo.",
    }
    if k in exact: return exact[k]
    if "stance" in k: return "Describe cuánto tiempo o qué proporción del ciclo permanece el pie apoyado en el suelo, según la estimación 2D."
    if "swing" in k: return "Describe la fase en la que el pie avanza sin apoyo, según la estimación 2D."
    if "knee" in k or "valgus" in k: return "Describe la alineación o movimiento proyectado de la rodilla en la imagen; no sustituye una medición 3D."
    if "pelvis" in k: return "Describe cómo se mueve o inclina la pelvis durante la marcha."
    if "trunk" in k or "shoulder" in k: return "Describe el movimiento del tronco/hombros y su coordinación con la pelvis."
    if "foot" in k or "rearfoot" in k: return "Describe la orientación proyectada del pie o retropié en la imagen; debe interpretarse con la calidad de visibilidad distal."
    if "quality" in k or "score" in k or "coherence" in k: return "Control interno que indica si el cálculo dispone de datos suficientemente coherentes para ser interpretado."
    return f"Parámetro calculado por PhysioSentinel: {label}. Consulte la nota técnica para conocer su alcance."


def _annex_interpretation(m):
    k=str(m.get("key", "")); q=str(m.get("quality", "") or ""); v=m.get("value")
    if any(t in q.lower() for t in ["no fiable","no calculable","requiere"]):
        return "No interpretar clínicamente: dato no disponible o calidad insuficiente."
    try: fv=float(v); finite=np.isfinite(fv)
    except Exception: finite=False; fv=np.nan
    if k=="step_to_step_similarity_index" and finite:
        return ("Consistencia temporal alta entre pasos consecutivos." if fv>=90 else "Consistencia temporal buena, con pequeñas diferencias entre pasos." if fv>=75 else "Variabilidad paso a paso moderada." if fv>=60 else "Variabilidad paso a paso elevada; revisar junto con vídeo y calidad.")
    if k=="regularity_cv" and finite:
        return "Variabilidad temporal baja/favorable." if fv<=5 else "Variabilidad temporal apreciable; contextualizar clínicamente."
    if k=="temporal_asymmetry_exp" and finite:
        return "Buena similitud temporal bilateral." if fv<=5 else "Existe diferencia temporal entre ambos lados; valorar magnitud y contexto."
    if k.startswith("com_"):
        return "Descriptor de estabilidad/transferencia lateral. Priorizar comparación intraindividual; no existe un umbral diagnóstico universal para este proxy 2D."
    if k.startswith("biomech_profile_index") and finite:
        return _biomech_classification(fv)+" (ICBF (Índice de Control Biomecánico Frontal) / ICS (Índice de Control Cinemático Sagital) 0.4 interno; dependiente de la vista y no diagnóstico)."
    if k.startswith("gait_index_global") and finite:
        return _biomech_classification(fv)+" (alias histórico; desde v29 corresponde al perfil biomecánico dependiente de la vista)."
    if q:
        return f"Interpretar con calidad: {q}."
    return "Dato descriptivo; interpretar junto con el resto del análisis y el contexto clínico."


def build_annex_dataframe(metrics):
    rows=[]
    for m in metrics or []:
        rows.append({
            "Grupo": _annex_group(m.get("key")),
            "Variable": m.get("label"),
            "Clave técnica": m.get("key"),
            "Qué significa para el paciente": _patient_explanation_for_metric(m),
            "Resultado": m.get("value"),
            "Unidad": m.get("unit"),
            "Valor / contexto de referencia": reference_text_for_metric(m.get("key","")),
            "Interpretación": _annex_interpretation(m),
            "Calidad": m.get("quality"),
            "Nota técnica": m.get("notes"),
        })
    return pd.DataFrame(rows)


# === V61 · Informe integrado + anexo maestro multiplanar/3D ===
def _v61_float(v):
    try:
        x=float(v)
        return x if np.isfinite(x) else None
    except Exception:
        return None


def _v61_metric_map(metrics):
    out={}
    for m in metrics or []:
        if isinstance(m,dict):
            out[str(m.get("key", ""))]=m
    return out


def _v61_find_metric(metrics, tokens):
    toks=[str(t).lower() for t in tokens]
    for m in metrics or []:
        if not isinstance(m,dict):
            continue
        hay=(str(m.get("key", ""))+" "+str(m.get("label", ""))).lower()
        if all(t in hay for t in toks):
            v=_v61_float(m.get("value"))
            if v is not None:
                return m,v
    return None,None


def _v61_fmt_metric(m,v,dec=2):
    if m is None or v is None: return None
    unit=str(m.get("unit", "") or "").strip()
    return f"{m.get('label',m.get('key','Variable'))}: {v:.{dec}f}{(' '+unit) if unit else ''}"


def _v61_integrated_text(metrics):
    """Integra 2D frontal, 2D lateral, ciclo biplanar y 3D sin inventar normalidad/patología."""
    ss=st.session_state
    lines=[]
    lines.append("VISIÓN INTEGRADA DE LOS RESULTADOS")
    lines.append("Este apartado integra los hallazgos espaciotemporales, la cinemática 2D frontal/posterior y lateral, la coordinación biplanar y, cuando existe, la información 3D. Debe interpretarse junto con la calidad de captura y las limitaciones metodológicas de cada nivel.")

    # Espaciotemporal: escoger métricas existentes sin asumir nombres rígidos.
    picks=[]
    for toks in (["cadencia"],["ciclo","duración"],["apoyo","%"],["oscilación","%"],["velocidad"]):
        m,v=_v61_find_metric(metrics,toks)
        txt=_v61_fmt_metric(m,v)
        if txt and txt not in picks: picks.append(txt)
    if picks:
        lines.append("\n1. Componente espaciotemporal. " + "; ".join(picks[:6]) + ".")

    # Frontal y lateral: informar cuántas variables reales existen y ejemplos numéricos.
    front=[]; lat=[]
    for m in metrics or []:
        if not isinstance(m,dict): continue
        key=(str(m.get("key", ""))+" "+str(m.get("label", ""))).lower()
        v=_v61_float(m.get("value"))
        if v is None: continue
        txt=_v61_fmt_metric(m,v)
        if not txt: continue
        if "lateral" in key or "sagital" in key:
            if len(lat)<8: lat.append(txt)
        elif any(t in key for t in ["pelvis","tronco","valgo","retrop","pie","com","frontal"]):
            if len(front)<8: front.append(txt)
    if front:
        lines.append("\n2. Plano frontal/posterior. La vista frontal aporta el control mediolateral y la organización de pelvis, tronco, rodillas, pies/retropié y COM (Centro de Masa estimado 2D). Variables representativas del registro: " + "; ".join(front[:5]) + ".")
    if lat:
        lines.append("\n3. Plano lateral/sagital. La cámara lateral complementa la vista frontal con la evolución sagital de los segmentos y articulaciones durante el ciclo. Variables representativas del registro: " + "; ".join(lat[:6]) + ".")

    v48=ss.get("biplanar_v48_profile") or {}
    summ=(v48.get("summary") or {}) if isinstance(v48,dict) else {}
    if summ:
        cyc=_v61_float(summ.get("v52_cycle_duration_s") or summ.get("front_cycle_s"))
        stance=_v61_float(summ.get("v52_stance_pct"))
        swing=_v61_float(summ.get("v52_swing_pct"))
        to=_v61_float(summ.get("v52_to_pct") or summ.get("v51_selected_to_pct"))
        dic=_v61_float(summ.get("ic_alignment_delta_ms"))
        q=str(summ.get("integration_confidence") or summ.get("v52_quality") or "No especificada")
        bits=[]
        if cyc is not None: bits.append(f"ciclo IC→IC {cyc:.2f} s")
        if stance is not None: bits.append(f"apoyo {stance:.1f} %")
        if to is not None: bits.append(f"TO {to:.1f} % del ciclo")
        if swing is not None: bits.append(f"oscilación {swing:.1f} %")
        if dic is not None: bits.append(f"discrepancia residual de IC {dic:.1f} ms")
        lines.append("\n4. Integración biplanar. Las dos cámaras se relacionan sobre el mismo ciclo físico, permitiendo interpretar conjuntamente qué ocurre en los planos frontal y sagital en cada fase. " + ("; ".join(bits)+". " if bits else "") + f"Calidad/confianza de integración: {q}. Esta integración 2D multiplanar no equivale por sí sola a una triangulación 3D métrica.")

    est=ss.get("v60_estimated_3d")
    if isinstance(est,dict) and isinstance(est.get("points3d"),pd.DataFrame) and not est["points3d"].empty:
        p=est["points3d"]; acc=p[p.get("accepted_3d",False)==True] if "accepted_3d" in p.columns else p
        lines.append(f"\n5. 3D estimado/no métrico. Disponible una representación espacial cualitativa fusionando ambas vistas ({acc['phase_pct'].nunique() if 'phase_pct' in acc.columns else 0} instantes; {acc['landmark'].nunique() if 'landmark' in acc.columns else 0} landmarks). Sus X/Y/Z están normalizadas: no deben emplearse como cm/mm ni para inferir ángulos articulares 3D cuantitativos.")
    else:
        lines.append("\n5. 3D estimado/no métrico. No consta una reconstrucción estimada disponible en el estado actual del análisis.")

    cal=ss.get("v56_triangulation")
    if isinstance(cal,dict) and isinstance(cal.get("points3d"),pd.DataFrame) and not cal["points3d"].empty:
        p=cal["points3d"]
        acc=p[p["accepted_3d"]==True] if "accepted_3d" in p.columns else p
        total=len(p); ok=len(acc); pct=(100*ok/total) if total else 0
        med=_v61_float(pd.to_numeric(acc.get("reproj_mean_px"),errors="coerce").median()) if "reproj_mean_px" in acc else None
        p95=_v61_float(pd.to_numeric(acc.get("reproj_mean_px"),errors="coerce").quantile(.95)) if "reproj_mean_px" in acc else None
        ang=_v61_float(pd.to_numeric(acc.get("ray_angle_deg"),errors="coerce").median()) if "ray_angle_deg" in acc else None
        bits=[f"{ok}/{total} puntos aceptados ({pct:.1f} %)"]
        if med is not None: bits.append(f"reproyección mediana {med:.2f} px")
        if p95 is not None: bits.append(f"P95 (percentil 95) {p95:.2f} px")
        if ang is not None: bits.append(f"ángulo de triangulación mediano {ang:.1f}°")
        lines.append("\n6. 3D calibrado/cuantitativo. Existe triangulación geométrica de ambas cámaras: " + "; ".join(bits) + ". Las coordenadas X/Y/Z solo se consideran cuantitativas dentro del modelo y unidad definidos por el perfil de calibración activo.")
    else:
        lines.append("\n6. 3D calibrado/cuantitativo. No consta triangulación calibrada disponible; por tanto no se atribuyen distancias ni coordenadas espaciales métricas a partir del 3D estimado.")

    lines.append("\nSÍNTESIS INTEGRADA. El resultado debe leerse como un único patrón de marcha: los parámetros temporales indican cuándo ocurren los eventos; la vista frontal/posterior describe el control mediolateral; la lateral describe la evolución sagital; y la capa biplanar relaciona ambos planos en el mismo ciclo. El 3D estimado añade una representación espacial cualitativa y el 3D calibrado, cuando está disponible y supera sus controles de calidad, añade información espacial cuantitativa. La interpretación final debe priorizar las variables con mejor calidad y concordancia y no convertir proxies 2D o coordenadas normalizadas en mediciones 3D métricas.")
    return "\n".join(lines)



def _v62_rehabilitation_text(metrics):
    """Convierte hallazgos observados en objetivos de rehabilitación trazables, sin inferir etiología."""
    ss=st.session_state
    candidates=[]

    def add(priority, domain, finding, objective, followup, caveat):
        if finding and objective:
            candidates.append({
                "priority": priority, "domain": domain, "finding": finding,
                "objective": objective, "followup": followup, "caveat": caveat,
            })

    # Selección basada exclusivamente en variables realmente calculadas.
    usable=[]
    for m in metrics or []:
        if not isinstance(m,dict):
            continue
        v=_v61_float(m.get("value"))
        if v is None:
            continue
        text=(str(m.get("key", ""))+" "+str(m.get("label", ""))).lower()
        usable.append((m,v,text))

    # Asimetría / diferencias entre lados: prioridad funcional alta cuando la propia métrica lo expresa.
    asym=[]
    for m,v,t in usable:
        if any(k in t for k in ["asimetr", "diferencia derecha", "diferencia izquierda", "left_right", "right_left", "simetr"]):
            asym.append((_v61_fmt_metric(m,v),abs(v),m))
    if asym:
        asym.sort(key=lambda x:x[1],reverse=True)
        vals="; ".join(x[0] for x in asym[:3] if x[0])
        add(1,"Simetría y organización bilateral",vals,
            "Reducir progresivamente las diferencias funcionales entre ambos miembros durante el ciclo, buscando una ejecución más reproducible y equilibrada sin imponer una simetría artificial.",
            "Reevaluar las mismas variables de asimetría/diferencia entre lados y su comportamiento ciclo a ciclo.",
            "La asimetría cinemática no identifica por sí sola su causa; contrastar fuerza, movilidad, dolor, sensibilidad y control motor en la exploración clínica.")

    # Variabilidad / consistencia.
    var=[]
    for m,v,t in usable:
        if any(k in t for k in ["variab", "consistencia", "desviación", "dispersion", "dispersión", "cv "]):
            var.append((_v61_fmt_metric(m,v),abs(v),m))
    if var:
        var.sort(key=lambda x:x[1],reverse=True)
        vals="; ".join(x[0] for x in var[:3] if x[0])
        add(1 if not candidates else 2,"Consistencia del patrón de marcha",vals,
            "Mejorar la reproducibilidad del patrón entre pasos y ciclos, manteniendo una estrategia estable durante la tarea de marcha.",
            "Seguir variabilidad/consistencia entre ciclos junto con cadencia, duración de ciclo y eventos IC/TO cuando estén disponibles.",
            "La variabilidad puede depender de velocidad, ayuda técnica, fatiga o contexto; comparar registros bajo condiciones equivalentes.")

    # Plano sagital: rodilla/cadera/tobillo y fase de oscilación/apoyo.
    sag=[]
    for m,v,t in usable:
        if ("lateral" in t or "sagital" in t) and any(k in t for k in ["rodilla","knee","cadera","hip","tobillo","ankle","flex","extens","rom"]):
            sag.append((_v61_fmt_metric(m,v),m))
    if sag:
        vals="; ".join(x[0] for x in sag[:4] if x[0])
        add(2,"Control sagital y progresión del miembro",vals,
            "Optimizar la coordinación sagital de cadera, rodilla y tobillo a lo largo de apoyo y oscilación, priorizando la fase en la que el registro muestre la mayor limitación o compensación.",
            "Reevaluar ROM (Rango de Movimiento), picos articulares y momento del ciclo en que aparecen, además de apoyo/oscilación e IC/TO.",
            "La cinemática describe el movimiento observado; no permite atribuir automáticamente el hallazgo a debilidad, rigidez o déficit neurológico específico.")

    # Plano frontal / pelvis / tronco / base.
    front=[]
    for m,v,t in usable:
        if any(k in t for k in ["pelvis","tronco","frontal","valgo","retrop","base","bos","mediolateral","com"]):
            if "lateral" not in t:
                front.append((_v61_fmt_metric(m,v),m))
    if front:
        vals="; ".join(x[0] for x in front[:4] if x[0])
        add(2,"Control frontal y estabilidad mediolateral",vals,
            "Mejorar el control mediolateral durante la transferencia de carga y el apoyo, coordinando pelvis, tronco y miembro inferior sin aumentar compensaciones en otros planos.",
            "Seguir las mismas variables frontales de pelvis/tronco/rodilla/pie, COM (Centro de Masa estimado 2D) y BOS (Base de Sustentación) que estén disponibles.",
            "Una desviación frontal no demuestra por sí sola déficit de un músculo concreto; la selección de ejercicios requiere exploración clínica complementaria.")

    # Temporal / apoyo-oscilación.
    temp=[]
    for m,v,t in usable:
        if any(k in t for k in ["cadencia","apoyo","oscilación","stance","swing","ciclo","contacto inicial","toe off","to "]):
            temp.append((_v61_fmt_metric(m,v),m))
    if temp:
        vals="; ".join(x[0] for x in temp[:4] if x[0])
        add(3,"Organización temporal del ciclo",vals,
            "Consolidar una secuencia temporal eficiente entre contacto inicial, apoyo, despegue y oscilación, adaptada a la capacidad funcional de la persona.",
            "Reevaluar cadencia, duración del ciclo, porcentajes de apoyo/oscilación e IC (Initial Contact / Contacto Inicial)–TO (Toe Off / Despegue del Pie).",
            "Los porcentajes temporales deben interpretarse junto con velocidad de marcha, ayuda técnica y objetivo funcional.")

    # Biplanar/3D como control de coherencia, no como diagnóstico causal.
    v48=ss.get("biplanar_v48_profile") or {}
    summ=(v48.get("summary") or {}) if isinstance(v48,dict) else {}
    if summ:
        q=str(summ.get("integration_confidence") or summ.get("v52_quality") or "No especificada")
        add(3,"Coordinación multiplanar",f"Integración frontal-lateral disponible; calidad/confianza: {q}.",
            "Favorecer un patrón coordinado entre los planos frontal y sagital, evitando mejorar una variable aislada a costa de una compensación en el otro plano.",
            "Comparar conjuntamente las curvas frontales y laterales sobre el mismo ciclo físico en las reevaluaciones.",
            "El 3D estimado es cualitativo/no métrico. Solo el 3D calibrado puede aportar coordenadas espaciales cuantitativas dentro de su modelo de calibración.")

    lines=[]
    lines.append("REHABILITACIÓN · OBJETIVOS DERIVADOS DEL ANÁLISIS DE LA MARCHA")
    lines.append("Los objetivos siguientes se derivan de los hallazgos observados en este registro y sirven para orientar la planificación y la reevaluación. No constituyen un diagnóstico etiológico ni prescriben por sí solos una técnica de tratamiento; deben contrastarse con la exploración clínica, los objetivos funcionales de la persona y el contexto asistencial.")
    if not candidates:
        lines.append("\nNo hay suficientes variables calculadas en el estado actual para formular objetivos específicos con trazabilidad. Se recomienda completar el análisis y definir los objetivos tras la exploración clínica.")
        return "\n".join(lines)

    # Máximo 5 objetivos para mantener priorización clínica legible.
    candidates=sorted(candidates,key=lambda x:x["priority"])[:5]
    for i,c in enumerate(candidates,1):
        pname={1:"PRINCIPAL",2:"SECUNDARIA",3:"SEGUIMIENTO"}.get(c["priority"],"SEGUIMIENTO")
        lines.append(f"\n{i}. PRIORIDAD {pname} · {c['domain']}")
        lines.append(f"Hallazgo que lo justifica: {c['finding']}")
        lines.append(f"Objetivo de rehabilitación: {c['objective']}")
        lines.append(f"Variables para seguimiento: {c['followup']}")
        lines.append(f"Precaución interpretativa: {c['caveat']}")

    lines.append("\nCRITERIO GENERAL DE REEVALUACIÓN. Repetir el análisis en condiciones comparables y valorar conjuntamente cambio cuantitativo, consistencia entre ciclos, repercusión funcional y presencia o ausencia de compensaciones. Una variación numérica aislada no debe etiquetarse automáticamente como mejoría o empeoramiento.")
    return "\n".join(lines)

def _v61_master_annex(metrics):
    base=build_annex_dataframe(metrics)
    extra=[]
    def add(group,key,label,value,unit,meaning,quality="Control interno",note=""):
        extra.append({"Grupo":group,"Variable":key,"Nombre clínico":label,"Qué mide / significado":meaning,"Resultado":value,"Unidad":unit,"Valor / contexto de referencia":"Sin rango normativo automático","Interpretación":"Descriptiva; integrar con el resto del análisis.","Calidad":quality,"Nota técnica":note})
    ss=st.session_state
    v48=ss.get("biplanar_v48_profile") or {}
    summ=(v48.get("summary") or {}) if isinstance(v48,dict) else {}
    for k,v in summ.items():
        if isinstance(v,(int,float,np.integer,np.floating)) and not isinstance(v,bool) and np.isfinite(float(v)):
            add("Biplanar",f"biplanar_{k}",f"Biplanar · {k}",float(v),"según variable","Parámetro del ciclo físico homólogo frontal-lateral.",str(summ.get("integration_confidence","Control interno")),"Procede de las capas V48–V53; no implica 3D métrico.")
    pm=v48.get("v52_phase_metrics") if isinstance(v48,dict) else None
    if isinstance(pm,pd.DataFrame) and not pm.empty:
        for _,r in pm.iterrows():
            var=str(r.get("variable",r.get("name","variable")))
            for c in pm.columns:
                if c in ["variable","name"]: continue
                val=_v61_float(r.get(c))
                if val is not None:
                    add("Biplanar · fases",f"v52_{var}_{c}",f"{var} · {c}",val,"según variable","Medición de la variable sobre apoyo/oscilación o eventos del ciclo clínico biplanar.",str(summ.get("v52_quality","Control interno")),"V52; conservar la unidad original de la variable fuente.")
    est=ss.get("v60_estimated_3d")
    if isinstance(est,dict) and isinstance(est.get("points3d"),pd.DataFrame) and not est["points3d"].empty:
        p=est["points3d"]; acc=p[p["accepted_3d"]==True] if "accepted_3d" in p else p
        add("3D estimado","v60_3d_estimated_available","3D estimado disponible",1,"bool","Existe representación 3D cualitativa fusionada de frontal+lateral.","Cualitativa","X/Y/Z normalizadas; NO métricas.")
        add("3D estimado","v60_3d_estimated_instants","Instantes 3D estimados",int(acc["phase_pct"].nunique()) if "phase_pct" in acc else 0,"instantes","Número de instantes del ciclo representados en 3D estimado.","Control interno","No expresa precisión geométrica.")
        for ax in ["X","Y","Z"]:
            add("3D estimado",f"v60_{ax.lower()}_normalized",f"Coordenada {ax} estimada",np.nan,"unidad normalizada",f"Coordenada {ax} de visualización espacial aproximada.","No métrica","No usar para cm/mm ni cinemática 3D cuantitativa.")
    cal=ss.get("v56_triangulation")
    if isinstance(cal,dict) and isinstance(cal.get("points3d"),pd.DataFrame) and not cal["points3d"].empty:
        p=cal["points3d"]; acc=p[p["accepted_3d"]==True] if "accepted_3d" in p else p
        total=len(p); ok=len(acc)
        add("3D calibrado","v56_points_accepted_pct","Puntos 3D aceptados",100*ok/total if total else np.nan,"%","Proporción de candidatos que supera los controles geométricos V56.","QC (Control de Calidad)","Triangulación calibrada.")
        if "reproj_mean_px" in acc:
            rr=pd.to_numeric(acc["reproj_mean_px"],errors="coerce")
            add("3D calibrado","v56_reproj_median_px","Error de reproyección mediano",rr.median(),"px","Concordancia entre el punto 3D reproyectado y las observaciones 2D.","QC (Control de Calidad)","Menor error geométrico implica mejor consistencia interna, sin rango clínico automático.")
            add("3D calibrado","v56_reproj_p95_px","Error de reproyección P95",rr.quantile(.95),"px","Percentil 95 del error de reproyección.","QC (Control de Calidad)","")
        if "ray_angle_deg" in acc:
            aa=pd.to_numeric(acc["ray_angle_deg"],errors="coerce")
            add("3D calibrado","v56_ray_angle_median_deg","Ángulo de triangulación mediano",aa.median(),"°","Geometría de intersección de los rayos de ambas cámaras.","QC (Control de Calidad)","")
        for ax in ["X","Y","Z"]:
            add("3D calibrado",f"v56_{ax.lower()}_calibrated",f"Coordenada {ax} calibrada",np.nan,"unidad de calibración",f"Coordenada espacial {ax} obtenida por triangulación calibrada.","Cuantitativa condicionada a calibración","La unidad física depende del perfil de calibración activo.")
    if extra:
        ex=pd.DataFrame(extra)
        if base is None or base.empty: return ex
        return pd.concat([base,ex],ignore_index=True,sort=False)
    return base


tabs = st.tabs([
    "1 · Vídeos", "2 · Calidad", "3 · Analizar marcha", "4 · Resultados 2D",
    "5 · Pacientes / Evolución", "6 · Redacción informe", "7 · Biplanar / Calibración",
    "8 · Exportar / Descargar", "9 · Ciclo de marcha",
    "10 · Capacidad funcional + Perfil biomecánico",
    "11 · Anexo · Diccionario de variables"
])

with tabs[0]:
    st.subheader("Carga temporal de vídeo")
    st.info("Los vídeos se usan únicamente para este análisis. No se suben a Supabase ni quedan guardados en el histórico.")
    if not is_two_cam:
        up1 = st.file_uploader(f"Vídeo {view.lower()}", type=["mp4","mov","avi","mkv"], key="uploader_video1")
        if up1:
            st.video(up1)
        up2 = None
    else:
        st.markdown("### Protocolo 2 cámaras")
        st.write("Cámara 1: frontal/posterior. Cámara 2: lateral. No es necesario que ambas vean al paciente durante todo el recorrido: Sentinel identifica la ventana en la que el mismo paciente es visible simultáneamente y usa solo esa zona para el análisis biplanar.")
        st.caption("Las cámaras pueden empezar a grabar en instantes diferentes. Si es posible, deja unos segundos de margen. La Versión 54 mantiene congelado el estimador temporal V47 y estima automáticamente el desfase temporal usando el movimiento del paciente y la ventana de visibilidad compartida; un evento común visible mejora la comprobación, pero no es obligatorio.")
        c1, c2 = st.columns(2)
        with c1:
            up1 = st.file_uploader("Cámara 1 · frontal/posterior", type=["mp4","mov","avi","mkv"], key="uploader_front")
            if up1:
                st.video(up1)
        with c2:
            up2 = st.file_uploader("Cámara 2 · lateral", type=["mp4","mov","avi","mkv"], key="uploader_side")
            if up2:
                st.video(up2)
        st.caption("Versión 54: RTMPose (Real-Time Multi-Person Pose Estimation / Estimación de Pose Multipersona en Tiempo Real) procesa ambas cámaras; Sentinel detecta el solapamiento visible, estima el desfase temporal y empareja ciclos del mismo instante. La reconstrucción 3D (tridimensional) y la cinética todavía no se ejecutan.")

    if st.button("Crear sesión temporal", type="primary", use_container_width=True):
        if up1 is None or (is_two_cam and up2 is None):
            st.error("Selecciona el/los vídeo(s) necesarios.")
        else:
            try:
                folder = create_temp_session(patient, record)
                p1 = folder / "videos" / f"cam01{Path(up1.name).suffix.lower() or '.mp4'}"
                save_upload(up1, p1)
                meta1 = video_metadata(p1)
                if not meta1:
                    raise RuntimeError("No puedo leer el vídeo de cámara 1.")
                p2 = None
                meta2 = None
                if up2 is not None:
                    p2 = folder / "videos" / f"cam02{Path(up2.name).suffix.lower() or '.mp4'}"
                    save_upload(up2, p2)
                    meta2 = video_metadata(p2)
                    if not meta2:
                        raise RuntimeError("No puedo leer el vídeo de cámara 2.")
                st.session_state.update({
                    "session_dir": str(folder), "video1_path": str(p1), "video2_path": str(p2) if p2 else None,
                    "meta1": meta1, "meta2": meta2, "mode": mode, "view": view,
                    "subject_mode": subject_mode,
                    "pose_done": False, "pose_raw_done": False, "subject_locked": False,
                    "pose_analysis_running": False, "pose_execution_id": None,
                    "subject_selection_cam1": None, "subject_selection_cam2": None,
                    "subject_preview_cam1": None, "subject_preview_cam2": None,
                    "multiperson_prepass_done": False, "multiperson_full_done": False,
                    "multiperson_prepass_dir": None, "multiperson_pose_dir": None,
                    "multiperson_samples_cam1": None, "multiperson_samples_cam2": None,
                    "tracking_info_cam1": None, "tracking_info_cam2": None,
                    "metrics_done": False, "temp_deleted": False,
                    "analysis_df": None, "analysis_df2": None,
                    "annotated_video_bytes": None, "annotated_video2_bytes": None,
                    "patient_code": st.session_state.get("patient_code", patient.strip()),
                    "record_name": st.session_state.get("record_name", record.strip()),
                    "patient_age": int(st.session_state.get("patient_age", 0) or 0),
                    "patient_sex": st.session_state.get("patient_sex", "No especificado"),
                    "record_date": st.session_state.get("record_date", datetime.now().date()),
                    "assistive_device": assistive_device, "frontal_orientation": frontal_orientation,
                    "sync_offset_auto_s": 0.0, "sync_offset_user_s": 0.0,
                    "sync_correlation": np.nan, "sync_quality": "No calculable",
                    "camera1_quality": None, "camera2_quality": None,
                    "cloud_session_id": None,
                    "selected_calibration_name": st.session_state.get("selected_calibration_name"),
                })
                cloud_ok = False
                if sb_ready():
                    try:
                        sid = sb_create_session(
                            st.session_state.get("patient_code", patient.strip()),
                            st.session_state.get("record_name", record.strip()),
                            mode, view, meta1,
                            assistive_device, frontal_orientation,
                            meta2=meta2,
                            calibration_profile_name=st.session_state.get("selected_calibration_name"),
                        )
                        st.session_state["cloud_session_id"] = sid
                        cloud_ok = True
                    except Exception as sb_e:
                        st.session_state["cloud_session_id"] = None
                        st.warning(f"La sesión temporal se ha creado, pero Supabase no pudo registrar esta sesión: {sb_e}")
                if cloud_ok:
                    st.success("Sesión temporal creada. Supabase conserva la sesión y su contexto; los vídeos permanecen solo en /tmp.")
                else:
                    st.success("Sesión temporal creada. Puedes analizar y ver resultados aunque Supabase no esté disponible.")
            except Exception as e:
                st.error(str(e))

with tabs[1]:
    st.subheader("Control de calidad de captura")
    meta1 = st.session_state.get("meta1")
    meta2 = st.session_state.get("meta2")
    if not meta1:
        st.info("Crea primero una sesión temporal.")
    else:
        if st.session_state.get("mode", mode).startswith("2 cámaras") and meta2:
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("### Cámara 1 · frontal/posterior")
                st.metric("FPS", f"{meta1['fps']:.2f}")
                st.metric("Duración", f"{meta1['duration']:.2f} s")
                st.write(f"Resolución: **{meta1['width']} × {meta1['height']}**")
            with c2:
                st.markdown("### Cámara 2 · lateral")
                st.metric("FPS", f"{meta2['fps']:.2f}")
                st.metric("Duración", f"{meta2['duration']:.2f} s")
                st.write(f"Resolución: **{meta2['width']} × {meta2['height']}**")
            fps_diff = abs(meta1["fps"] - meta2["fps"])
            dur_diff = abs(meta1["duration"] - meta2["duration"])
            a, b = st.columns(2)
            a.metric("Diferencia FPS", f"{fps_diff:.3f}")
            b.metric("Diferencia de duración", f"{dur_diff:.2f} s")
            if fps_diff <= 0.5:
                st.success("Las frecuencias de imagen son suficientemente próximas para la preparación 3D.")
            else:
                st.warning("Las cámaras tienen FPS diferentes. Se puede analizar 2D, pero la reconstrucción 3D requerirá remuestreo/sincronización cuidadosa.")
            if dur_diff > 2.0:
                st.warning("La duración difiere más de 2 s. Revisa que ambas grabaciones cubran el mismo ensayo.")
        else:
            a,b,c,d = st.columns(4)
            a.metric("FPS", f"{meta1['fps']:.1f}")
            b.metric("Duración", f"{meta1['duration']:.1f} s")
            c.metric("Resolución", f"{meta1['width']} × {meta1['height']}")
            d.metric("Orientación", meta1['orientation'])
        if st.session_state.get("assistive_device", "Sin ayuda") != "Sin ayuda":
            st.info(f"Marcha con ayuda técnica: **{st.session_state.get('assistive_device')}**. Tras RTMPose se calculará la visibilidad por cámara y región corporal.")

with tabs[2]:
    st.subheader("Analizar marcha")
    if not st.session_state.get("session_dir"):
        st.info("Crea primero una sesión temporal.")
    elif st.session_state.get("temp_deleted"):
        st.info("Los archivos temporales ya fueron eliminados después de guardar los resultados.")
    else:
        current_two = st.session_state.get("mode", "").startswith("2 cámaras")
        manual_mode = str(st.session_state.get("subject_mode","")).startswith("Selección manual")
        st.write("Motor: **Pose2Sim + RTMPose · Body_with_feet (HALPE26)**")
        st.caption("V74 · Seguimiento multipersona + máscara rectilínea global: el paciente se mantiene mientras siga siendo identificable; solo se excluyen los frames realmente ambiguos durante cruces/oclusiones.")
        if manual_mode:
            st.info("**Modo multipersona V74:** primero se detectan los sujetos en 15 frames. Tras elegir al paciente, el análisis completo mantiene su identidad incluso si otra persona pasa cerca. Solo los frames en los que realmente no puede distinguirse con seguridad qué pose pertenece al paciente se excluyen de métricas y gráficas; después se permite una reacquisición rápida y controlada.")
        if current_two:
            st.write("Se realizará detección de pose en **cam01 frontal/posterior y cam02 lateral** y después se estimará su desfase temporal.")

        _pose_busy, _pose_busy_info = _pose_lock_info()
        if _pose_busy:
            _started = (_pose_busy_info or {}).get("started_at", "")
            st.warning(
                "Ya hay una estimación de pose ejecutándose en esta instancia. "
                "Para evitar saturar la CPU, V68 no permite iniciar otra simultáneamente."
                + (f" Inicio: {_started}." if _started else "")
            )

        # V68 · flujo multipersona en dos etapas: preselección rápida -> análisis selectivo.
        if manual_mode:
            if not st.session_state.get("multiperson_prepass_done"):
                if st.button("1 · Detectar sujetos (rápido)", type="primary", use_container_width=True,
                             disabled=bool(_pose_busy or st.session_state.get("pose_analysis_running"))):
                    _pose_token=None
                    try:
                        session_dir=Path(st.session_state.session_dir)
                        _pose_token,_busy_info=_acquire_pose_lock(session_dir)
                        if not _pose_token:
                            raise RuntimeError("Ya existe otro análisis de pose en ejecución.")
                        st.session_state["pose_analysis_running"]=True
                        pre_dir=session_dir / "_multiperson_prepass_v68"
                        shutil.rmtree(pre_dir, ignore_errors=True)
                        (pre_dir/"videos").mkdir(parents=True, exist_ok=True)

                        samples1=_sample_frame_indices(st.session_state.meta1, n=15)
                        samples1=_make_sparse_video(st.session_state.video1_path, pre_dir/"videos"/"cam01.mp4", samples1)
                        st.session_state["multiperson_samples_cam1"]=samples1
                        if current_two:
                            samples2=_sample_frame_indices(st.session_state.meta2, n=15)
                            samples2=_make_sparse_video(st.session_state.video2_path, pre_dir/"videos"/"cam02.mp4", samples2)
                            st.session_state["multiperson_samples_cam2"]=samples2

                        with st.spinner("Preselección multipersona: buscando todos los sujetos en 15 frames representativos..."):
                            cfg=prepare_config(pre_dir, multi_person=True, det_frequency=1)
                            run_pose2sim(cfg)
                        st.session_state["multiperson_prepass_dir"]=str(pre_dir)

                        pre_json1=find_pose_json_dir(pre_dir,"cam01")
                        if not pre_json1:
                            raise RuntimeError("No encuentro resultados de la preselección en cámara 1.")
                        sel1=scan_subject_candidates(pre_json1)
                        if not sel1 or not sel1.get("candidates"):
                            raise RuntimeError("No se detectaron sujetos suficientemente visibles en la preselección rápida.")
                        sel1=_selection_to_original_frame(sel1, samples1)
                        st.session_state.subject_selection_cam1=sel1
                        st.session_state.subject_preview_cam1=render_subject_preview(st.session_state.video1_path, sel1)
                        st.session_state["prepass_json_cam1"]=str(pre_json1)

                        if current_two:
                            pre_json2=find_pose_json_dir(pre_dir,"cam02")
                            if not pre_json2:
                                raise RuntimeError("No encuentro resultados de la preselección en cámara 2.")
                            sel2=scan_subject_candidates(pre_json2)
                            if not sel2 or not sel2.get("candidates"):
                                raise RuntimeError("No se detectaron sujetos suficientemente visibles en cámara 2.")
                            sel2=_selection_to_original_frame(sel2, st.session_state.get("multiperson_samples_cam2") or [])
                            st.session_state.subject_selection_cam2=sel2
                            st.session_state.subject_preview_cam2=render_subject_preview(st.session_state.video2_path, sel2)
                            st.session_state["prepass_json_cam2"]=str(pre_json2)

                        st.session_state["multiperson_prepass_done"]=True
                        st.success(f"Preselección completada: {len(sel1['candidates'])} sujeto(s) visibles en cámara 1. Elige al paciente antes del análisis completo.")
                    except Exception as e:
                        st.error(f"Error en la preselección: {e}")
                        with st.expander("Detalles técnicos"):
                            st.code(traceback.format_exc())
                    finally:
                        if _pose_token:
                            _release_pose_lock(_pose_token)
                        st.session_state["pose_analysis_running"]=False

            if st.session_state.get("multiperson_prepass_done") and not st.session_state.get("multiperson_full_done"):
                st.markdown("### Selecciona el paciente")
                sel1=st.session_state.get("subject_selection_cam1")
                choice1=None
                if sel1:
                    if st.session_state.get("subject_preview_cam1"):
                        st.image(st.session_state.subject_preview_cam1,
                                 caption=f"Cámara 1 · frame original {sel1['frame']} · sujetos de izquierda a derecha",
                                 use_container_width=True)
                    labels1=[c["label"] for c in sel1["candidates"]]
                    choice1=st.radio("Paciente en cámara 1", labels1, key="manual_subject_choice_cam1_v68", horizontal=True)

                choice2=None; sel2=st.session_state.get("subject_selection_cam2")
                if current_two and sel2:
                    if st.session_state.get("subject_preview_cam2"):
                        st.image(st.session_state.subject_preview_cam2,
                                 caption=f"Cámara 2 · frame original {sel2['frame']} · selecciona a la misma persona física",
                                 use_container_width=True)
                    labels2=[c["label"] for c in sel2["candidates"]]
                    choice2=st.radio("Paciente en cámara 2", labels2, key="manual_subject_choice_cam2_v68", horizontal=True)

                st.caption("V74: la selección manual inicializa directamente la identidad y una trayectoria espacial esperada. La presencia del acompañante no corta por sí sola el seguimiento; solo se excluyen frames realmente ambiguos y la reacquisición no queda bloqueada por una pérdida breve.")
                if st.button("2 · Analizar paciente seleccionado", type="primary", use_container_width=True,
                             disabled=bool(_pose_busy or st.session_state.get("pose_analysis_running") or not choice1)):
                    _pose_token=None
                    try:
                        session_dir=Path(st.session_state.session_dir)
                        _pose_token,_busy_info=_acquire_pose_lock(session_dir)
                        if not _pose_token:
                            raise RuntimeError("Ya existe otro análisis de pose en ejecución.")
                        st.session_state["pose_analysis_running"]=True
                        work_dir=session_dir / "_subject_pose_v68"
                        shutil.rmtree(work_dir, ignore_errors=True)
                        (work_dir/"videos").mkdir(parents=True, exist_ok=True)

                        boxes1=_tracked_sparse_boxes(Path(st.session_state.prepass_json_cam1), sel1, choice1,
                                                     st.session_state.get("multiperson_samples_cam1") or [])
                        _build_masked_subject_video(st.session_state.video1_path, work_dir/"videos"/"cam01.mp4", boxes1)
                        if current_two:
                            if not choice2:
                                raise RuntimeError("Selecciona también el paciente en cámara 2.")
                            boxes2=_tracked_sparse_boxes(Path(st.session_state.prepass_json_cam2), sel2, choice2,
                                                         st.session_state.get("multiperson_samples_cam2") or [])
                            _build_masked_subject_video(st.session_state.video2_path, work_dir/"videos"/"cam02.mp4", boxes2)

                        with st.spinner("Analizando únicamente la región dinámica del paciente seleccionado..."):
                            cfg=prepare_config(work_dir, multi_person=True, det_frequency=4)
                            run_pose2sim(cfg)

                        json1=find_pose_json_dir(work_dir,"cam01")
                        if not json1:
                            raise RuntimeError("Pose2Sim terminó pero no encuentro los JSON selectivos de cam01.")
                        st.session_state.pose_json1=str(json1)
                        anchor1, person1, map1 = _resolve_fullrun_subject(json1, sel1, choice1)
                        df1, track1 = load_pose_dataframe_tracked(json1, anchor1, person1, prior_boxes=boxes1)
                        track1.update(map1)
                        if df1.empty:
                            raise RuntimeError("No se pudo mantener una trayectoria fiable del paciente seleccionado en cam01.")
                        if float(track1.get("identity_continuity_pct", 0.0) or 0.0) < 45.0:
                            raise RuntimeError("La continuidad de identidad del paciente en cam01 es demasiado baja; no se aceptan mezclas con otro sujeto.")
                        st.session_state.analysis_df=df1
                        st.session_state.subject_tracking_info_cam1=track1
                        q1=camera_pose_quality(df1); st.session_state.camera1_quality=q1
                        st.session_state.selected_subject_label_cam1=choice1

                        if current_two:
                            json2=find_pose_json_dir(work_dir,"cam02")
                            if not json2:
                                raise RuntimeError("Pose2Sim terminó pero no encuentro los JSON selectivos de cam02.")
                            st.session_state.pose_json2=str(json2)
                            anchor2, person2, map2 = _resolve_fullrun_subject(json2, sel2, choice2)
                            df2, track2 = load_pose_dataframe_tracked(json2, anchor2, person2, prior_boxes=boxes2)
                            track2.update(map2)
                            if df2.empty:
                                raise RuntimeError("No se pudo mantener una trayectoria fiable del paciente seleccionado en cam02.")
                            if float(track2.get("identity_continuity_pct", 0.0) or 0.0) < 45.0:
                                raise RuntimeError("La continuidad de identidad del paciente en cam02 es demasiado baja; no se aceptan mezclas con otro sujeto.")
                            st.session_state.analysis_df2=df2
                            st.session_state.subject_tracking_info_cam2=track2
                            q2=camera_pose_quality(df2); st.session_state.camera2_quality=q2
                            st.session_state.selected_subject_label_cam2=choice2
                            meta1=st.session_state.meta1; meta2=st.session_state.meta2
                            off,corr,sq,w1,w2=estimate_biplanar_sync(df1,meta1["fps"],df2,meta2["fps"])
                            st.session_state.biplanar_window_cam1=w1; st.session_state.biplanar_window_cam2=w2
                            st.session_state.sync_offset_auto_s=off; st.session_state.sync_correlation=corr; st.session_state.sync_quality=sq

                        st.session_state["multiperson_pose_dir"]=str(work_dir)
                        st.session_state["multiperson_full_done"]=True
                        st.session_state.pose_raw_done=True
                        st.session_state.pose_done=True
                        st.session_state.subject_locked=True
                        _write_pose_completion_marker(session_dir,{"version":"71","manual_mode":True,"completed_at":datetime.now().isoformat(timespec="seconds"),"choice_cam1":choice1,"choice_cam2":choice2})
                        st.success(f"Análisis selectivo completado: {len(df1)} frames útiles de cámara 1." + (f" Cámara 2: {len(df2)} frames útiles." if current_two else ""))
                        st.info("Abre 4 · Resultados 2D para revisar y calcular. El vídeo original no se ha recortado ni reescalado; el enmascarado solo se usó como entrada de pose.")
                    except Exception as e:
                        st.error(f"Error durante el análisis selectivo: {e}")
                        with st.expander("Detalles técnicos"):
                            st.code(traceback.format_exc())
                    finally:
                        if _pose_token:
                            _release_pose_lock(_pose_token)
                        st.session_state["pose_analysis_running"]=False

        else:
            _analysis_clicked = False
            if not st.session_state.get("pose_raw_done"):
                _analysis_clicked = st.button(
                    "▶ Analizar marcha",
                    type="primary",
                    use_container_width=True,
                    disabled=bool(_pose_busy or st.session_state.get("pose_analysis_running")),
                )

            if _analysis_clicked:
                _pose_token = None
                try:
                    session_dir = Path(st.session_state.session_dir)
                    marker=_pose_completion_marker(session_dir)
                    if marker.exists() and st.session_state.get("pose_json1"):
                        st.info("V68 detectó que esta sesión ya dispone de una pose completada; se reutilizan los resultados y no se vuelve a lanzar RTMPose.")
                        st.session_state.pose_raw_done=True; st.session_state.pose_done=True
                    else:
                        _pose_token, _busy_info = _acquire_pose_lock(session_dir)
                        if not _pose_token:
                            raise RuntimeError("Ya existe otro análisis de pose en ejecución.")
                        st.session_state["pose_analysis_running"] = True
                        st.session_state["pose_execution_id"] = _pose_token
                        try:
                            with st.spinner("Detectando pose con Pose2Sim/RTMPose..."):
                                cfg = prepare_config(session_dir)
                                run_pose2sim(cfg)
                        finally:
                            _release_pose_lock(_pose_token)
                            _pose_token = None
                            st.session_state["pose_analysis_running"] = False
                        json1 = find_pose_json_dir(session_dir, "cam01")
                        if not json1:
                            raise RuntimeError("Pose2Sim terminó pero no encuentro los JSON de cam01.")
                        st.session_state.pose_json1 = str(json1)
                        json2 = None
                        if current_two:
                            json2 = find_pose_json_dir(session_dir, "cam02")
                            if not json2:
                                raise RuntimeError("Pose2Sim terminó pero no encuentro los JSON de cam02.")
                            st.session_state.pose_json2 = str(json2)
                        df1 = load_pose_dataframe(json1)
                        if df1.empty:
                            raise RuntimeError("No se pudieron leer keypoints HALPE26 de cam01.")
                        st.session_state.analysis_df = df1
                        q1 = camera_pose_quality(df1); st.session_state.camera1_quality = q1
                        if current_two:
                            df2 = load_pose_dataframe(json2)
                            if df2.empty:
                                raise RuntimeError("No se pudieron leer keypoints HALPE26 de cam02.")
                            st.session_state.analysis_df2 = df2
                            q2 = camera_pose_quality(df2); st.session_state.camera2_quality = q2
                            meta1 = st.session_state.meta1; meta2 = st.session_state.meta2
                            off, corr, sq, w1, w2 = estimate_biplanar_sync(df1, meta1["fps"], df2, meta2["fps"])
                            st.session_state.biplanar_window_cam1 = w1; st.session_state.biplanar_window_cam2 = w2
                            st.session_state.sync_offset_auto_s = off; st.session_state.sync_correlation = corr; st.session_state.sync_quality = sq
                            st.success(f"Pose completada en dos cámaras: cam01 {len(df1)} frames útiles · cam02 {len(df2)} frames útiles.")
                        else:
                            st.session_state.analysis_df2 = None
                            st.success(f"Pose completada: {len(df1)} frames útiles.")
                        st.session_state.pose_raw_done = True; st.session_state.pose_done = True; st.session_state.subject_locked = True
                        _write_pose_completion_marker(session_dir,{"version":"71","manual_mode":False,"completed_at":datetime.now().isoformat(timespec="seconds")})
                        st.info("Abre 4 · Resultados 2D, revisa el desfase/segmento válido y calcula.")
                except Exception as e:
                    st.error(f"Error durante el análisis: {e}")
                    with st.expander("Detalles técnicos"):
                        st.code(traceback.format_exc())
                finally:
                    if _pose_token:
                        _release_pose_lock(_pose_token)
                    st.session_state["pose_analysis_running"] = False

        # Segunda etapa exclusivamente multipersona: elección humana + bloqueo de identidad.
        if manual_mode and st.session_state.get("pose_raw_done") and not st.session_state.get("subject_locked"):
            st.markdown("### Selección manual del sujeto biomecánico")
            st.warning("Selecciona al **paciente**, no al terapeuta/acompañante. La app conservará esa identidad y rechazará frames ambiguos en lugar de cambiar de persona.")

            sel1=st.session_state.get("subject_selection_cam1")
            if sel1:
                st.markdown("#### Cámara 1")
                if st.session_state.get("subject_preview_cam1"):
                    st.image(st.session_state.subject_preview_cam1, caption=f"Frame de selección {sel1['frame']} · sujetos ordenados de izquierda a derecha", use_container_width=True)
                labels1=[c["label"] for c in sel1["candidates"]]
                choice1=st.radio("Paciente en cámara 1", labels1, key="manual_subject_choice_cam1", horizontal=True)
            else:
                choice1=None

            choice2=None
            sel2=st.session_state.get("subject_selection_cam2")
            if current_two and sel2:
                st.markdown("#### Cámara 2")
                if st.session_state.get("subject_preview_cam2"):
                    st.image(st.session_state.subject_preview_cam2, caption=f"Frame de selección {sel2['frame']} · selecciona a la misma persona física", use_container_width=True)
                labels2=[c["label"] for c in sel2["candidates"]]
                choice2=st.radio("Paciente en cámara 2", labels2, key="manual_subject_choice_cam2", horizontal=True)

            if st.button("🔒 Bloquear sujeto y preparar análisis biomecánico", type="primary", use_container_width=True):
                try:
                    c1=next(c for c in sel1["candidates"] if c["label"]==choice1)
                    df1,t1=load_pose_dataframe_tracked(Path(st.session_state.pose_json1), sel1["frame"], c1["person_index"])
                    if df1.empty or len(df1)<10:
                        raise RuntimeError("El seguimiento del sujeto seleccionado en cam01 no produce suficientes frames fiables.")
                    t1["selected_label"] = choice1
                    st.session_state.selected_subject_label_cam1 = choice1
                    st.session_state.analysis_df=df1
                    st.session_state.tracking_info_cam1=t1
                    q1=camera_pose_quality(df1)
                    st.session_state.camera1_quality=q1

                    if current_two:
                        c2=next(c for c in sel2["candidates"] if c["label"]==choice2)
                        df2,t2=load_pose_dataframe_tracked(Path(st.session_state.pose_json2), sel2["frame"], c2["person_index"])
                        if df2.empty or len(df2)<10:
                            raise RuntimeError("El seguimiento del sujeto seleccionado en cam02 no produce suficientes frames fiables.")
                        t2["selected_label"] = choice2
                        st.session_state.selected_subject_label_cam2 = choice2
                        st.session_state.analysis_df2=df2
                        st.session_state.tracking_info_cam2=t2
                        q2=camera_pose_quality(df2)
                        st.session_state.camera2_quality=q2
                        meta1=st.session_state.meta1; meta2=st.session_state.meta2
                        off,corr,sq,w1,w2=estimate_biplanar_sync(df1,meta1["fps"],df2,meta2["fps"])
                        st.session_state.biplanar_window_cam1=w1
                        st.session_state.biplanar_window_cam2=w2
                        st.session_state.sync_offset_auto_s=off
                        # V54 corregida: la heurística NO modifica sync_offset_user_s
                        st.session_state.sync_correlation=corr
                        st.session_state.sync_quality=sq
                        ready,reasons=readiness_3d(st.session_state.mode,q1,q2,sq,st.session_state.get("selected_calibration_name"))
                        if sb_ready() and st.session_state.get("cloud_session_id"):
                            sb_update_session_3d(
                                st.session_state.cloud_session_id,
                                sync_offset_s=off,
                                sync_correlation=corr if np.isfinite(corr) else None,
                                sync_quality=sq,
                                calibration_profile_name=st.session_state.get("selected_calibration_name"),
                                ready_3d=bool(ready),
                            )
                    else:
                        st.session_state.analysis_df2=None

                    st.session_state.subject_locked=True
                    st.session_state.pose_done=True
                    st.success("🔒 Sujeto biomecánico bloqueado. La otra persona queda excluida del análisis.")
                    st.write(
                        f"Continuidad de identidad cam01: **{t1.get('identity_continuity_pct',np.nan):.1f}%** · "
                        f"frames excluidos: **{t1.get('frames_excluded',0)}** · calidad: **{t1.get('quality','')}**."
                    )
                    if current_two:
                        st.write(
                            f"Continuidad de identidad cam02: **{t2.get('identity_continuity_pct',np.nan):.1f}%** · "
                            f"frames excluidos: **{t2.get('frames_excluded',0)}** · calidad: **{t2.get('quality','')}**."
                        )
                    st.info("Abre **4 · Resultados 2D** y calcula el intervalo válido. Las métricas se obtendrán únicamente del sujeto seleccionado.")
                except Exception as e:
                    st.error(f"No se pudo bloquear el sujeto: {e}")
                    with st.expander("Detalles técnicos"):
                        st.code(traceback.format_exc())

        elif manual_mode and st.session_state.get("subject_locked"):
            t1=st.session_state.get("tracking_info_cam1") or {}
            st.success("🔒 Sujeto biomecánico bloqueado.")
            if t1:
                st.caption(
                    f"Cam01 · continuidad identidad {t1.get('identity_continuity_pct',np.nan):.1f}% · "
                    f"excluidos {t1.get('frames_excluded',0)}/{t1.get('frames_total',0)} frames · "
                    f"cambios de identidad evitados {t1.get('switches_prevented',0)}."
                )

with tabs[3]:
    st.subheader("Resultados 2D complementarios")
    df1 = st.session_state.get("analysis_df")
    df2 = st.session_state.get("analysis_df2")
    meta1 = st.session_state.get("meta1")
    meta2 = st.session_state.get("meta2")
    current_two = st.session_state.get("mode", "").startswith("2 cámaras")

    if (df1 is None or not st.session_state.get("pose_done") or not meta1) and not st.session_state.get("metrics_done"):
        st.info("Ejecuta primero **Analizar marcha**.")

    if df1 is not None and meta1:
        if current_two and df2 is not None and meta2:
            st.markdown("### Sincronización de cámaras")
            auto = float(st.session_state.get("sync_offset_auto_s", 0.0))
            corr = st.session_state.get("sync_correlation", np.nan)
            sq = st.session_state.get("sync_quality", "No calculable")
            st.caption("Convención: un valor positivo significa que el mismo instante físico aparece más tarde en cam02. Versión 47 separa reloj físico, estimador ciego y fase biomecánica.")
            st.write(f"Heurística preliminar de movimiento: **{auto:+.3f} s** · correlación **{fmt(corr,2)}** · calidad **{sq}**")
            st.caption("Esta heurística NO se interpreta como ground truth del reloj: dos vistas diferentes pueden presentar señales corporales con distinto desfase de fase.")
            st.caption("V54 CORREGIDA · La heurística preliminar es solo informativa: nunca modifica el desfase operativo. El campo parte de 0,00 s.")
            w1=st.session_state.get("biplanar_window_cam1"); w2=st.session_state.get("biplanar_window_cam2")
            if w1 and w2:
                st.write(f"Visibilidad del paciente · frontal/posterior: **{w1['start_s']:.2f}–{w1['end_s']:.2f} s** · lateral: **{w2['start_s']:.2f}–{w2['end_s']:.2f} s**")
                overlap_start=max(w1['start_s'], w2['start_s']-auto)
                overlap_end=min(w1['end_s'], w2['end_s']-auto)
                overlap_d=max(0.0,overlap_end-overlap_start)
                st.metric("Ventana biplanar detectada por heurística", f"{overlap_d:.2f} s", help="Ventana preliminar calculada con la heurística de movimiento. En modo validación, el reloj físico conocido prevalece.")
            st.caption("FPS (Frames Per Second / Fotogramas por Segundo). La sincronización física y la alineación de eventos biomecánicos son conceptos distintos.")

            # V47 · modo de validación con reloj conocido. Útil para datasets sincronizados
            # o adquisiciones con trigger/hardware externo. No altera el modo clínico normal.
            validation_known = st.checkbox(
                "Modo de validación · desfase físico conocido",
                value=bool(st.session_state.get("v46_validation_known", False)),
                help="Actívalo sólo si conoces el desfase real entre cámaras por hardware, trigger o ground truth del dataset. La fase de la marcha se audita, pero no puede mover el reloj físico.",
            )
            st.session_state.v46_validation_known = bool(validation_known)
            gt_offset = st.number_input(
                "Desfase físico conocido cam02 vs cam01 (s)",
                min_value=-15.0, max_value=15.0,
                value=float(st.session_state.get("v46_ground_truth_offset_s", 0.0)),
                step=0.04,
                disabled=not validation_known,
                help="Para cámaras realmente sincronizadas que comparten el mismo inicio temporal del recorte, usa 0,000 s.",
            )
            st.session_state.v46_ground_truth_offset_s = float(gt_offset)
            if validation_known:
                st.info(f"Reloj físico fijado por referencia conocida: **{float(gt_offset):+.3f} s**. La fase biomecánica se calculará por separado y no reajustará este valor.")

            dur1, dur2 = float(meta1["duration"]), float(meta2["duration"])
            max_sync = max(2.0, min(15.0, max(dur1, dur2)))

            # V54 CORREGIDA · migración única de la semántica del reloj operativo.
            # Evita arrastrar -0.96 (u otro valor heurístico) desde una sesión anterior.
            if st.session_state.get("_v54_operational_offset_migrated") is not True:
                st.session_state.sync_offset_user_s = 0.0
                st.session_state["_v54_operational_offset_migrated"] = True

            seed = float(gt_offset) if validation_known else float(st.session_state.get("sync_offset_user_s", 0.0))
            if not np.isfinite(seed): seed = 0.0
            seed = float(np.clip(seed, -max_sync, max_sync))
            sync_user = st.number_input(
                "Desfase operativo cam02 respecto a cam01 (s)",
                min_value=float(-max_sync), max_value=float(max_sync),
                value=seed,
                step=0.01,
                disabled=bool(validation_known),
                help="En uso clínico puede corregirse manualmente. En modo validación queda bloqueado al ground truth para no contaminar el reloj físico.",
            )
            if validation_known:
                sync_user = float(gt_offset)
            st.session_state.sync_offset_user_s = float(sync_user)
            common_start = max(0.0, -float(sync_user))
            common_end = min(dur1, dur2 - float(sync_user))
            w1=st.session_state.get("biplanar_window_cam1"); w2=st.session_state.get("biplanar_window_cam2")
            if w1:
                common_start=max(common_start,float(w1["start_s"])); common_end=min(common_end,float(w1["end_s"]))
            if w2:
                common_start=max(common_start,float(w2["start_s"])-float(sync_user)); common_end=min(common_end,float(w2["end_s"])-float(sync_user))
            if common_end <= common_start + 1.0:
                st.error("No queda un intervalo temporal común suficiente con el desfase seleccionado.")
            else:
                start_s, end_s = st.slider(
                    "Intervalo válido común (tiempo de cam01)",
                    float(round(common_start,2)), float(round(common_end,2)),
                    (float(round(common_start,2)), float(round(common_end,2))),
                    step=0.02,
                )
                st.caption(f"Cam02 analizará aproximadamente {start_s+sync_user:.2f}–{end_s+sync_user:.2f} s para representar el mismo intervalo físico.")
                if st.button("Calcular ambas vistas, guardar histórico y eliminar vídeos", type="primary", use_container_width=True):
                    try:
                        f1a, f1b = int(round(start_s*meta1["fps"])), int(round(end_s*meta1["fps"]))
                        s2a, s2b = start_s + sync_user, end_s + sync_user
                        f2a, f2b = int(round(s2a*meta2["fps"])), int(round(s2b*meta2["fps"]))
                        m_front, chart_front, seg_front = compute_metrics(
                            df1, float(meta1["fps"]), f1a, f1b, "Frontal/posterior", st.session_state.get("assistive_device","Sin ayuda"), st.session_state.get("scale_cm_per_px",0.0)
                        )
                        m_lat, chart_lat, seg_lat = compute_metrics(
                            df2, float(meta2["fps"]), f2a, f2b, "Lateral", st.session_state.get("assistive_device","Sin ayuda"), 0.0
                        )
                        metrics = prefix_metrics(m_front, "front", "Frontal/posterior") + prefix_metrics(m_lat, "lateral", "Lateral")
                        metrics += tracking_metrics(st.session_state.get("tracking_info_cam1"), "front", "Frontal/posterior")
                        metrics += tracking_metrics(st.session_state.get("tracking_info_cam2"), "lateral", "Lateral")
                        metrics += [
                            {"key":"sync_offset_cam02_s","label":"Desfase cam02 vs cam01","value":sync_user,"unit":"s","quality":sq,"notes":"Sincronización temporal experimental; valor positivo = evento más tardío en cam02."},
                            {"key":"sync_correlation","label":"Correlación de sincronización automática","value":corr,"unit":"r","quality":sq,"notes":"Correlación heurística de movimiento corporal vertical entre cámaras."},
                        ]
                        q1 = st.session_state.get("camera1_quality") or camera_pose_quality(df1)
                        q2 = st.session_state.get("camera2_quality") or camera_pose_quality(df2)
                        ready, reasons = readiness_3d(st.session_state.mode, q1, q2, sq, st.session_state.get("selected_calibration_name"))
                        metrics.append({"key":"ready_3d_flag","label":"Preparación para triangulación 3D","value":1.0 if ready else 0.0,"unit":"bool","quality":"Preparado" if ready else "Pendiente","notes":"; ".join(reasons) if reasons else "Dos cámaras, visibilidad suficiente, sincronización aceptable y perfil de calibración seleccionado."})
                        # Versión 33 · detectar ciclos sobre TODA la ventana visible de cada cámara.
                        # La selección de la ventana común se aplica DESPUÉS al emparejar ciclos.
                        snap_front = _make_biplanar_cycle_snapshot(df1, float(meta1["fps"]), "Frontal/posterior")
                        snap_lat = _make_biplanar_cycle_snapshot(df2, float(meta2["fps"]), "Lateral")
                        st.session_state.cycle_seg_front = snap_front["seg"] if snap_front else seg_front.copy()
                        st.session_state.cycle_seg_lateral = snap_lat["seg"] if snap_lat else seg_lat.copy()
                        st.session_state.cycle_fps_front = float(meta1["fps"])
                        st.session_state.cycle_fps_lateral = float(meta2["fps"])
                        st.session_state.cycle_support_left_front = snap_front["left"] if snap_front else None
                        st.session_state.cycle_support_right_front = snap_front["right"] if snap_front else None
                        st.session_state.cycle_support_left_lateral = snap_lat["left"] if snap_lat else None
                        st.session_state.cycle_support_right_lateral = snap_lat["right"] if snap_lat else None
                        st.session_state.biplanar_lateral_event_info = (snap_lat.get("event_info") or {}) if snap_lat else {}
                        full_front_cycles = len(_cycles_from_snapshot(snap_front)) if snap_front else 0
                        full_lateral_cycles = len(_cycles_from_snapshot(snap_lat)) if snap_lat else 0
                        pairs,pair_q,pair_diag=match_biplanar_cycles(
                            snap_front,snap_lat,float(sync_user),
                            physical_window=(float(start_s),float(end_s)),
                            physical_clock_override_s=(float(gt_offset) if validation_known else None)
                        )
                        st.session_state.biplanar_cycle_pairs=pairs
                        st.session_state.biplanar_cycle_quality=pair_q
                        st.session_state.biplanar_v40_event_pairs=pair_diag.get("v40_event_pairs_data",[]) or []
                        st.session_state.biplanar_v40_triplets=pair_diag.get("v40_triplet_data",[]) or []
                        st.session_state.biplanar_v41_rescue_rows=pair_diag.get("v41_rescue_rows",[]) or []
                        st.session_state.biplanar_v41_triplets=pair_diag.get("v41_triplet_data",[]) or []
                        st.session_state.biplanar_v42_triplet_audit=pair_diag.get("v42_triplet_audit",[]) or []
                        st.session_state.biplanar_v42_candidate_audit=pair_diag.get("v42_candidate_audit",[]) or []
                        st.session_state.biplanar_v43_closure_cycles=pair_diag.get("v43_closure_cycle_data",[]) or []
                        st.session_state.biplanar_v44_chains=pair_diag.get("v44_chain_data",[]) or []
                        st.session_state.biplanar_v44_chain_audit=pair_diag.get("v44_chain_audit",[]) or []
                        st.session_state.biplanar_v45_near_prediction_candidates=pair_diag.get("v45_near_prediction_candidates",[]) or []
                        st.session_state.biplanar_v45_status=pair_diag.get("v45_status","")
                        st.session_state.biplanar_v45_reason=pair_diag.get("v45_reason","")
                        st.session_state.biplanar_v45_last_accepted_side=pair_diag.get("v45_last_accepted_side","")
                        st.session_state.biplanar_v45_last_accepted_frame=pair_diag.get("v45_last_accepted_frame",np.nan)
                        st.session_state.biplanar_v45_expected_next_side=pair_diag.get("v45_expected_next_side","")
                        st.session_state.biplanar_v46_physical_clock_offset_s=float(pair_diag.get("v46_physical_clock_offset_s",np.nan))
                        st.session_state.biplanar_v46_biomechanical_phase_offset_s=float(pair_diag.get("v46_biomechanical_phase_offset_s",np.nan))
                        st.session_state.biplanar_v46_phase_minus_physical_s=float(pair_diag.get("v46_phase_minus_physical_s",np.nan))
                        st.session_state.biplanar_v46_clock_source=str(pair_diag.get("v46_physical_clock_source",""))
                        pair_mae=float(np.mean([abs(p["delta_s"]) for p in pairs])) if pairs else np.nan
                        # V50 · construir un único ciclo físico homólogo y auditar TO sin tocar el reloj V47.
                        v48_profile=_build_v48_biplanar_profile(
                            pairs, st.session_state.cycle_seg_front, st.session_state.cycle_seg_lateral,
                            float(meta1["fps"]), float(meta2["fps"]), pair_q
                        )
                        st.session_state.biplanar_v48_profile=v48_profile
                        # V54 CORREGIDA 3 · actualizar preparación geométrica
                        # en el MISMO punto en que nace el ciclo homólogo V48.
                        if isinstance(v48_profile, dict):
                            _v54s_live = v48_profile.get("summary") or {}
                            _v54_cycle_live = bool(
                                isinstance(_v54s_live, dict)
                                and _v54s_live.get("selected_pair_index") is not None
                                and str(_v54s_live.get("front_side", "")).strip()
                                and str(_v54s_live.get("lateral_side", "")).strip()
                            )
                            st.session_state.v54_geometry_readiness = {
                                "version": "54-corregida-3",
                                "purpose": "Preparación geométrica pre-3D",
                                "sync_layer": "V47 congelada",
                                "homologous_cycle_available": _v54_cycle_live,
                                "homologous_pair_index": _v54s_live.get("selected_pair_index") if _v54_cycle_live else None,
                                "front_side": _v54s_live.get("front_side") if _v54_cycle_live else None,
                                "lateral_side": _v54s_live.get("lateral_side") if _v54_cycle_live else None,
                                "integration_confidence": _v54s_live.get("integration_confidence") if _v54_cycle_live else None,
                                "ic_alignment_delta_ms": _v54s_live.get("ic_alignment_delta_ms") if _v54_cycle_live else None,
                                "front_cycle_s": _v54s_live.get("front_cycle_s") if _v54_cycle_live else None,
                                "lateral_cycle_s": _v54s_live.get("lateral_cycle_s") if _v54_cycle_live else None,
                                "variables_front": _v54s_live.get("variables_front") if _v54_cycle_live else None,
                                "variables_lateral": _v54s_live.get("variables_lateral") if _v54_cycle_live else None,
                                "intrinsics_cam01": "pendiente",
                                "intrinsics_cam02": "pendiente",
                                "extrinsics_cam01_cam02": "pendiente",
                                "simultaneous_fov_overlap": "por verificar",
                                "triangulation_3d_enabled": False,
                                "source": "biplanar_v48_profile actualizado tras cálculo",
                            }
                        if isinstance(v48_profile,dict):
                            _v48s=v48_profile.get("summary") or {}
                            metrics += [
                                {"key":"v48_homologous_cycle_available","label":"V48 · ciclo físico homólogo integrado","value":1.0,"unit":"bool","quality":_v48s.get("integration_confidence","No calculable"),"notes":"1 = existe un ciclo emparejado frontal-lateral normalizado a 0–100%. Integración 2D multiplanar; no 3D."},
                                {"key":"v48_selected_pair_index","label":"V48 · par de ciclo seleccionado","value":float(_v48s.get("selected_pair_index",np.nan)),"unit":"par","quality":"Control interno","notes":"Par con menor combinación de error IC y diferencia de duración entre los pares ya aceptados por V47."},
                                {"key":"v48_ic_alignment_delta_ms","label":"V48 · discrepancia IC entre planos","value":float(_v48s.get("ic_alignment_delta_ms",np.nan)),"unit":"ms","quality":pair_q,"notes":"Diferencia temporal residual del IC (Initial Contact / Contacto Inicial) del ciclo homólogo tras aplicar el reloj físico."},
                                {"key":"v48_cycle_duration_delta_ms","label":"V48 · diferencia de duración del ciclo","value":float(_v48s.get("cycle_duration_delta_ms",np.nan)),"unit":"ms","quality":pair_q,"notes":"Diferencia entre duración IC→IC frontal y lateral del mismo ciclo físico."},
                                {"key":"v48_front_to_pct","label":"V48 · TO frontal dentro del ciclo","value":float(_v48s.get("front_to_pct",np.nan)),"unit":"% ciclo","quality":"Experimental","notes":"TO (Toe Off / Despegue del Pie) estimado en vista frontal/posterior."},
                                {"key":"v48_lateral_to_pct","label":"V49 · TO lateral dentro del ciclo","value":float(_v48s.get("lateral_to_pct",np.nan)),"unit":"% ciclo","quality":"Experimental","notes":"TO (Toe Off / Despegue del Pie) estimado en vista lateral."},
                                {"key":"v49_to_concordance_delta_pct","label":"V49 · discrepancia TO frontal-lateral","value":float(_v48s.get("to_delta_pct",np.nan)),"unit":"puntos % ciclo","quality":str(_v48s.get("to_concordance_quality","No calculable")),"notes":"Diferencia absoluta entre TO estimado en ambos planos. ≤8 puntos: concordante; 8–15: limitada; >15: discrepante/no confirmado."},
                                {"key":"v49_biplanar_to_confirmed_pct","label":"V49 · TO biplanar confirmado","value":float(_v48s.get("biplanar_to_confirmed_pct",np.nan)),"unit":"% ciclo","quality":str(_v48s.get("to_concordance_quality","No calculable")),"notes":"Solo se informa cuando los TO de ambas vistas concuerdan dentro de 8 puntos porcentuales del ciclo; si no, permanece NaN para evitar falsa confirmación biplanar."},
                                {"key":"v49_front_variables_exported","label":"V49 · variables frontales integradas","value":float(_v48s.get("variables_front",0)),"unit":"variables","quality":"Control interno","notes":"Número de curvas frontales/posteriores realmente incorporadas al ciclo normalizado."},
                                {"key":"v49_lateral_variables_exported","label":"V49 · variables laterales integradas","value":float(_v48s.get("variables_lateral",0)),"unit":"variables","quality":"Control interno","notes":"Número de curvas sagitales/laterales realmente incorporadas al ciclo normalizado."},
                                {"key":"v50_front_to_candidate_pct","label":"V50 · TO candidato frontal multiseñal","value":float(_v48s.get("v50_front_to_candidate_pct",np.nan)),"unit":"% ciclo","quality":str(_v48s.get("v50_front_to_quality","No calculable")),"notes":"Onset distal multiseñal 2D en frontal/posterior; evidencia secundaria por geometría."},
                                {"key":"v50_lateral_to_candidate_pct","label":"V50 · TO candidato lateral multiseñal","value":float(_v48s.get("v50_lateral_to_candidate_pct",np.nan)),"unit":"% ciclo","quality":str(_v48s.get("v50_lateral_to_quality","No calculable")),"notes":"Onset distal multiseñal 2D en lateral; usa avance relativo del pie, ascenso distal y velocidad."},
                                {"key":"v50_to_candidate_delta_pct","label":"V50 · discrepancia entre candidatos TO","value":float(_v48s.get("v50_candidate_delta_pct",np.nan)),"unit":"puntos % ciclo","quality":str(_v48s.get("v50_to_quality","No calculable")),"notes":"Diferencia entre los nuevos candidatos V50 de ambas vistas."},
                                {"key":"v50_biplanar_to_confirmed_pct","label":"V50 · TO biplanar confirmado","value":float(_v48s.get("v50_biplanar_to_confirmed_pct",np.nan)),"unit":"% ciclo","quality":str(_v48s.get("v50_to_quality","No calculable")),"notes":"Solo se informa si los dos detectores multiseñal convergen dentro de 8 puntos porcentuales y superan calidad mínima."},
                                {"key":"v51_selected_to_pct","label":"V51 · TO seleccionado jerárquico","value":float(_v48s.get("v51_selected_to_pct",np.nan)),"unit":"% ciclo","quality":str(_v48s.get("v51_to_quality","No calculable")),"notes":"TO gobernado por la vista lateral/sagital; la frontal/posterior actúa como evidencia secundaria y no desplaza el evento primario."},
                                {"key":"v51_to_confirmed","label":"V51 · TO confirmado jerárquicamente","value":float(_v48s.get("v51_to_confirmed",0.0)),"unit":"bool","quality":str(_v48s.get("v51_to_quality","No calculable")),"notes":"1 = evidencia suficiente según jerarquía dependiente de perspectiva; 0 = lateral probable/no confirmado. Umbrales internos experimentales."},
                                {"key":"v51_front_support_at_lateral_to","label":"V51 · respaldo frontal en instante TO lateral","value":float(_v48s.get("v51_front_support_at_lateral_to",np.nan)),"unit":"ratio","quality":"Experimental","notes":"Activación frontal relativa a su pico, evaluada exactamente en el TO propuesto por la vista lateral."},
                                {"key":"v51_selected_to_uncertainty_ms","label":"V51 · resolución temporal práctica del TO","value":float(_v48s.get("v51_selected_to_uncertainty_ms",np.nan)),"unit":"ms","quality":"Resolución temporal","notes":"±1 frame de la cámara lateral expresado en milisegundos; no es un intervalo de confianza estadístico."},
                                {"key":"v52_stance_pct","label":"V52 · fase de apoyo","value":float(_v48s.get("v52_stance_pct",np.nan)),"unit":"% ciclo","quality":str(_v48s.get("v52_quality","No calculable")),"notes":"Porcentaje IC→TO del ciclo físico homólogo usando el TO V51; no se recalcula el evento."},
                                {"key":"v52_swing_pct","label":"V52 · fase de oscilación","value":float(_v48s.get("v52_swing_pct",np.nan)),"unit":"% ciclo","quality":str(_v48s.get("v52_quality","No calculable")),"notes":"Porcentaje TO→siguiente IC del mismo ciclo físico homólogo."},
                                {"key":"v52_stance_duration_s","label":"V52 · duración de apoyo","value":float(_v48s.get("v52_stance_duration_s",np.nan)),"unit":"s","quality":str(_v48s.get("v52_quality","No calculable")),"notes":"Duración temporal derivada del ciclo V47 y TO V51."},
                                {"key":"v52_swing_duration_s","label":"V52 · duración de oscilación","value":float(_v48s.get("v52_swing_duration_s",np.nan)),"unit":"s","quality":str(_v48s.get("v52_quality","No calculable")),"notes":"Duración temporal derivada del ciclo V47 y TO V51."},
                                {"key":"v52_variables_total","label":"V52 · variables integradas en ciclo clínico","value":float(_v48s.get("v52_variables_total",0)),"unit":"variables","quality":"Control interno","notes":"Número de curvas 2D frontales y laterales descritas por fase de apoyo/oscilación."},
                            ]
                        else:
                            metrics.append({"key":"v48_homologous_cycle_available","label":"V48 · ciclo físico homólogo integrado","value":0.0,"unit":"bool","quality":"No calculable","notes":"No se encontró un par de ciclo con datos suficientes para construir curvas biplanares normalizadas."})
                        # V39 · exportación temporal completa: IC de ciclo + candidatos laterales auditables.
                        _master_off=float(pair_diag.get("master_clock_offset_s",sync_user))
                        _final_off=float(pair_diag.get("final_offset_s",_master_off))
                        _event_metrics=[]
                        for _view,_snap,_off in [("front",snap_front,0.0),("lateral",snap_lat,_final_off)]:
                            if _snap:
                                _seen=set()
                                for _side,_obj in [("L",_snap.get("left")),("R",_snap.get("right"))]:
                                    if _obj:
                                        _frames=list(np.asarray(_obj.get("ic",[]),dtype=float))+list(np.asarray(_obj.get("next_ic",[]),dtype=float))
                                        _n=0
                                        for _fr in _frames:
                                            if not np.isfinite(_fr): continue
                                            _key=(_side,int(round(_fr)))
                                            if _key in _seen: continue
                                            _seen.add(_key); _n+=1
                                            _fps=float(_snap.get("fps",0) or 0)
                                            if _fps>0:
                                                _raw=float(_fr/_fps)
                                                _physical=float(_raw-_off)
                                                _event_metrics.append({
                                                    "key":f"v39_event_ic_{_view}_{_side}_{_n:02d}_physical_s",
                                                    "label":f"V39 IC {_view} {_side} {_n:02d} · tiempo físico",
                                                    "value":_physical,"unit":"s","quality":"Control interno",
                                                    "notes":f"Frame={int(round(_fr))}; tiempo cámara={_raw:.6f} s; offset aplicado={_off:.6f} s; incluye IC terminales next_ic."
                                                })
                        # Auditoría completa de candidatos laterales: aceptado/rechazado, score y motivo.
                        if snap_lat:
                            _fps=float(snap_lat.get("fps",0) or 0)
                            _audit=(snap_lat.get("event_info") or {}).get("candidate_audit",[]) or []
                            for _n,_ev in enumerate(_audit,1):
                                _fr=float(_ev.get("frame",np.nan))
                                if _fps>0 and np.isfinite(_fr):
                                    _raw=_fr/_fps
                                    _physical=_raw-_final_off
                                    _status=str(_ev.get("status","desconocido"))
                                    _score=float(_ev.get("score",np.nan))
                                    _reason=str(_ev.get("reason",""))
                                    _event_metrics.append({
                                        "key":f"v39_lateral_candidate_{_n:02d}_physical_s",
                                        "label":f"V39 candidato IC lateral {_n:02d} · {_ev.get('side','?')} · {_status}",
                                        "value":float(_physical),"unit":"s","quality":"Control interno",
                                        "notes":f"Frame={int(round(_fr))}; tiempo cámara={_raw:.6f} s; score={_score:.4f}; estado={_status}; motivo={_reason}"
                                    })
                                    _event_metrics.append({
                                        "key":f"v39_lateral_candidate_{_n:02d}_score",
                                        "label":f"V39 candidato IC lateral {_n:02d} · puntuación",
                                        "value":_score,"unit":"score","quality":"Control interno",
                                        "notes":f"Lado={_ev.get('side','?')}; estado={_status}; frame={int(round(_fr))}."
                                    })
                        metrics += _event_metrics
                        metrics += [
                            {"key":"v47_independent_clock_estimate_s","label":"V47 · estimación ciega independiente del reloj (V38)","value":float(pair_diag.get("v47_independent_clock_estimate_s",np.nan)),"unit":"s","quality":"Validación" if pair_diag.get("v46_ground_truth_active",0) else "Control interno","notes":"Estimación interna calculada sin imponer el ground truth. En modo validación se compara contra la referencia externa, pero NO gobierna el emparejamiento."},
                            {"key":"v47_clock_estimator_error_ms","label":"V47 · error firmado del estimador de reloj","value":1000.0*float(pair_diag.get("v47_clock_estimator_error_s",np.nan)),"unit":"ms","quality":"Referencia" if pair_diag.get("v46_ground_truth_active",0) else "No aplicable","notes":"Estimación ciega V38 - ground truth. A diferencia del error V46 anclado, este valor NO es cero por construcción."},
                            {"key":"v47_clock_estimator_abs_error_ms","label":"V47 · error absoluto del estimador de reloj","value":1000.0*float(pair_diag.get("v47_clock_estimator_abs_error_s",np.nan)),"unit":"ms","quality":str(pair_diag.get("v47_validation_class","No aplicable")),"notes":"Magnitud del error temporal independiente frente al ground truth."},
                            {"key":"v47_clock_estimator_error_practical_frames","label":"V47 · error del reloj en frames prácticos","value":float(pair_diag.get("v47_clock_estimator_error_practical_frames",np.nan)),"unit":"frames","quality":str(pair_diag.get("v47_validation_class","No aplicable")),"notes":"Error absoluto dividido por la duración de frame de la cámara más lenta; evita atribuir precisión sub-frame no demostrada."},
                            {"key":"v47_front_frame_ms","label":"V47 · duración de frame frontal/posterior","value":float(pair_diag.get("v47_front_frame_ms",np.nan)),"unit":"ms","quality":"Control interno","notes":"Derivada del FPS real frontal/posterior."},
                            {"key":"v47_lateral_frame_ms","label":"V47 · duración de frame lateral","value":float(pair_diag.get("v47_lateral_frame_ms",np.nan)),"unit":"ms","quality":"Control interno","notes":"Derivada del FPS real lateral."},
                            {"key":"v47_temporal_uncertainty_ms","label":"V47 · incertidumbre temporal conservadora combinada","value":float(pair_diag.get("v47_temporal_uncertainty_ms",np.nan)),"unit":"ms","quality":"Control interno","notes":"Suma de medio frame de cada cámara; referencia conservadora para interpretar diferencias entre marcas temporales."},
                            {"key":"v47_practical_resolution_ms","label":"V47 · resolución temporal práctica","value":float(pair_diag.get("v47_practical_resolution_ms",np.nan)),"unit":"ms","quality":"Control interno","notes":"Duración de frame de la cámara más lenta."},
                            {"key":"v46_validation_ground_truth_active","label":"V46 · referencia temporal conocida activa","value":float(pair_diag.get("v46_ground_truth_active",0)),"unit":"bool","quality":"Control interno","notes":"1 = el reloj físico se fijó por hardware/trigger/dataset y la fase biomecánica no pudo reajustarlo."},
                            {"key":"v46_physical_clock_offset_s","label":"V46 · desfase del reloj físico","value":float(pair_diag.get("v46_physical_clock_offset_s",np.nan)),"unit":"s","quality":"Control interno","notes":f"Fuente={pair_diag.get('v46_physical_clock_source','')}. Este valor representa el reloj, no la fase de la marcha."},
                            {"key":"v46_ground_truth_offset_s","label":"V46 · ground truth del desfase físico","value":float(pair_diag.get("v46_ground_truth_offset_s",np.nan)),"unit":"s","quality":"Referencia" if pair_diag.get("v46_ground_truth_active",0) else "No aplicable","notes":"Sólo disponible cuando el usuario declara una referencia temporal externa conocida."},
                            {"key":"v46_physical_clock_error_ms","label":"V46 · error del reloj físico frente a ground truth","value":1000.0*float(pair_diag.get("v46_physical_clock_error_s",np.nan)),"unit":"ms","quality":"Referencia" if pair_diag.get("v46_ground_truth_active",0) else "No aplicable","notes":"Diferencia reloj físico usado - ground truth. En modo validación anclado debe ser 0 por construcción; sirve para auditar que ninguna fase lo haya movido."},
                            {"key":"v46_biomechanical_phase_offset_s","label":"V46 · desfase de fase biomecánica","value":float(pair_diag.get("v46_biomechanical_phase_offset_s",np.nan)),"unit":"s","quality":pair_diag.get("phase_quality","No calculable"),"notes":"Alineación periódica de eventos/secuencia de marcha. No debe interpretarse como desfase físico de cámaras."},
                            {"key":"v46_phase_minus_physical_ms","label":"V46 · diferencia fase biomecánica - reloj físico","value":1000.0*float(pair_diag.get("v46_phase_minus_physical_s",np.nan)),"unit":"ms","quality":pair_diag.get("phase_quality","No calculable"),"notes":"Cuantifica cuánto se separa la alineación biomecánica del reloj físico. Es un indicador diagnóstico de sesgo entre vistas/eventos."},
                            {"key":"v46_matching_offset_s","label":"V46 · desfase usado para emparejar ciclos","value":float(pair_diag.get("v46_matching_offset_s",np.nan)),"unit":"s","quality":pair_q,"notes":"Con ground truth activo coincide con el reloj físico; sin ground truth conserva el comportamiento clínico V45."},
                            {"key":"v45_second_cycle_evaluable","label":"V45 · segundo ciclo lateral evaluable","value":float(pair_diag.get("v45_second_cycle_evaluable",0)),"unit":"bool","quality":pair_q,"notes":str(pair_diag.get("v45_status",""))+" · "+str(pair_diag.get("v45_reason",""))},
                            {"key":"v45_coverage_sufficient","label":"V45 · cobertura lateral suficiente para siguiente IC","value":float(pair_diag.get("v45_coverage_sufficient",0)),"unit":"bool","quality":"Control interno","notes":"1 = el siguiente IC esperado cae dentro de la ventana lateral visible con margen temporal."},
                            {"key":"v45_predicted_next_ic_frame","label":"V45 · frame lateral esperado del siguiente IC","value":float(pair_diag.get("v45_predicted_next_ic_frame",np.nan)),"unit":"frame","quality":"Control interno","notes":f"Lado esperado={pair_diag.get('v45_expected_next_side','')}; último IC aceptado={pair_diag.get('v45_last_accepted_side','')} frame {pair_diag.get('v45_last_accepted_frame',np.nan)}."},
                            {"key":"v45_predicted_next_ic_camera_s","label":"V45 · tiempo cámara esperado del siguiente IC","value":float(pair_diag.get("v45_predicted_next_ic_camera_s",np.nan)),"unit":"s","quality":"Control interno","notes":"Predicción = último IC lateral aceptado + periodo de paso robusto."},
                            {"key":"v45_visibility_end_frame","label":"V45 · último frame de visibilidad lateral","value":float(pair_diag.get("v45_visibility_end_frame",np.nan)),"unit":"frame","quality":"Control interno","notes":"Final de la ventana de visibilidad activa usada por el análisis biplanar."},
                            {"key":"v45_visibility_margin_ms","label":"V45 · margen de cobertura tras el IC esperado","value":1000*float(pair_diag.get("v45_visibility_margin_s",np.nan)),"unit":"ms","quality":"Control interno","notes":"Positivo = el IC esperado cae antes del final visible; negativo = la cámara termina antes."},
                            {"key":"v44_chain_count","label":"V44 · cadenas biplanares consecutivas","value":float(pair_diag.get("v44_chain_count",0)),"unit":"cadenas","quality":pair_q,"notes":"Secuencia A-B-A-B-A con dos ciclos consecutivos y un IC compartido obligatorio."},
                            {"key":"v44_chained_cycles","label":"V44 · ciclos consecutivos homólogos","value":float(pair_diag.get("v44_chained_cycles",0)),"unit":"ciclos","quality":pair_q,"notes":"Número de ciclos contenidos en la cadena biplanar validada."},
                            {"key":"v44_second_cycle_homolog","label":"V44 · segundo ciclo lateral homólogo","value":float(pair_diag.get("v44_second_cycle_homolog",0)),"unit":"bool","quality":pair_q,"notes":"1 = se ha validado una cadena de dos ciclos consecutivos; el IC final del primero es exactamente el IC inicial del segundo."},
                            {"key":"v44_direct_chains","label":"V44 · cadenas hipótesis directa","value":float(pair_diag.get("v44_direct_chains",0)),"unit":"cadenas","quality":"Control interno","notes":"Hipótesis L↔L / R↔R."},
                            {"key":"v44_swapped_chains","label":"V44 · cadenas hipótesis L/R invertida","value":float(pair_diag.get("v44_swapped_chains",0)),"unit":"cadenas","quality":"Control interno","notes":"Hipótesis L↔R / R↔L."},
                            {"key":"v43_closure_cycles","label":"V43 · ciclos homólogos por cierre","value":float(pair_diag.get("v43_closure_cycles",0)),"unit":"ciclos","quality":pair_q,"notes":"Ciclos aceptados por coincidencia de IC inicial/final, alternancia anatómica, duración global y periodicidad lateral; el IC intermedio puede quedar marcado como discrepante."},
                            {"key":"v43_second_lateral_homolog","label":"V43 · segundo ciclo lateral homólogo por cierre","value":float(pair_diag.get("v43_second_lateral_homolog",0)),"unit":"bool","quality":pair_q,"notes":"1 = existen al menos dos ciclos laterales distintos aceptados por cierre de ciclo."},
                            {"key":"v41_rescue_attempted","label":"V41 · búsqueda periódica del tercer IC","value":float(pair_diag.get("v41_rescue_attempted",0)),"unit":"bool","quality":"Control interno","notes":"1 = se intentó recuperar un tercer IC (Initial Contact / Contacto Inicial) real observado a partir de periodicidad y alternancia; nunca se inventan eventos."},
                            {"key":"v41_rescued_events","label":"V41 · IC reales recuperados por periodicidad","value":float(pair_diag.get("v41_rescued_events",0)),"unit":"eventos","quality":pair_q,"notes":"Eventos observados que estaban fuera del emparejamiento V40 estricto pero cumplen predicción temporal, alternancia y compatibilidad biplanar."},
                            {"key":"v41_triplet_cycles","label":"V41 · ciclos biplanares validados por triplete","value":float(pair_diag.get("v41_triplet_cycles",0)),"unit":"ciclos","quality":pair_q,"notes":"IC ipsilateral inicial → IC contralateral → siguiente IC ipsilateral, incluyendo rescate periódico solo cuando toda la secuencia es coherente."},
                            {"key":"v41_second_lateral_homolog","label":"V41 · segundo ciclo lateral con homólogo frontal","value":float(pair_diag.get("v41_second_lateral_homolog",0)),"unit":"bool","quality":pair_q,"notes":"1 = al menos dos ciclos laterales distintos tienen triplete frontal homólogo validado."},
                            {"key":"v41_predicted_step_period_ms","label":"V41 · periodo de paso usado para predicción","value":1000.0*float(pair_diag.get("v41_predicted_step_period_s",np.nan)),"unit":"ms","quality":pair_q,"notes":"Mediana robusta de los periodos frontal y lateral estimados por V39."},
                            {"key":"v41_rescue_tolerance_ms","label":"V41 · ventana máxima de rescate periódico","value":1000.0*float(pair_diag.get("v41_rescue_tolerance_s",np.nan)),"unit":"ms","quality":"Control interno","notes":"Ventana local alrededor del instante predicho, adaptada a FPS (Frames Per Second / Fotogramas por Segundo), con límites conservadores. No amplía la tolerancia general de sincronización."},
                            {"key":"v40_front_fps","label":"FPS frontal/posterior","value":float(pair_diag.get("v40_front_fps",np.nan)),"unit":"FPS","quality":"Control interno","notes":"FPS (Frames Per Second / Fotogramas por Segundo) real usado para convertir frames frontales a tiempo."},
                            {"key":"v40_lateral_fps","label":"FPS lateral","value":float(pair_diag.get("v40_lateral_fps",np.nan)),"unit":"FPS","quality":"Control interno","notes":"FPS (Frames Per Second / Fotogramas por Segundo) real usado para convertir frames laterales a tiempo."},
                            {"key":"v40_front_frame_ms","label":"Duración de un frame frontal","value":1000.0*float(pair_diag.get("v40_front_frame_s",np.nan)),"unit":"ms","quality":"Control interno","notes":"Resolución temporal nominal de la cámara frontal."},
                            {"key":"v40_lateral_frame_ms","label":"Duración de un frame lateral","value":1000.0*float(pair_diag.get("v40_lateral_frame_s",np.nan)),"unit":"ms","quality":"Control interno","notes":"Resolución temporal nominal de la cámara lateral."},
                            {"key":"v40_temporal_uncertainty_ms","label":"Incertidumbre temporal conservadora por cuantización","value":1000.0*float(pair_diag.get("v40_temporal_uncertainty_s",np.nan)),"unit":"ms","quality":"Control interno","notes":"Suma de ± medio frame de ambas cámaras; evita interpretar coincidencias por debajo de la resolución física del vídeo."},
                            {"key":"v40_practical_resolution_ms","label":"Resolución temporal práctica mínima","value":1000.0*float(pair_diag.get("v40_practical_resolution_s",np.nan)),"unit":"ms","quality":"Control interno","notes":"Duración de un frame de la cámara más lenta."},
                            {"key":"v40_event_pairs","label":"Pares de pasos sincronizados V40","value":float(pair_diag.get("v40_event_pairs",0)),"unit":"eventos","quality":pair_q,"notes":"Correspondencias IC construidas antes de formar ciclos, respetando orden temporal y tolerancia adaptada a FPS."},
                            {"key":"v40_triplet_cycles","label":"Ciclos biplanares por triplete V40","value":float(pair_diag.get("v40_triplet_cycles",0)),"unit":"ciclos","quality":pair_q,"notes":"Ciclos reconstruidos como IC ipsilateral inicial → IC contralateral → siguiente IC ipsilateral."},
                            {"key":"v40_second_lateral_homolog","label":"Segundo ciclo lateral con homólogo frontal V40","value":float(pair_diag.get("v40_second_lateral_homolog",0)),"unit":"bool","quality":pair_q,"notes":"1 = al menos dos ciclos laterales distintos forman tripletes biplanares válidos."},
                            {"key":"v40_event_mae_ms","label":"MAE temporal de pasos V40","value":1000.0*float(pair_diag.get("v40_event_mae_s",np.nan)),"unit":"ms","quality":pair_q,"notes":"MAE (Mean Absolute Error / Error Absoluto Medio) entre pasos frontal-lateral sincronizados."},
                            {"key":"v40_event_sequence_error_ms","label":"Error de secuencia de pasos V40","value":1000.0*float(pair_diag.get("v40_event_sequence_error_s",np.nan)),"unit":"ms","quality":pair_q,"notes":"Diferencia media entre intervalos consecutivos de pasos en ambas cámaras."},
                            {"key":"v40_direct_event_pairs","label":"Pares de pasos V40 · hipótesis directa","value":float(pair_diag.get("v40_direct_event_pairs",0)),"unit":"eventos","quality":"Control interno","notes":"Correspondencias L↔L/R↔R."},
                            {"key":"v40_swapped_event_pairs","label":"Pares de pasos V40 · hipótesis invertida","value":float(pair_diag.get("v40_swapped_event_pairs",0)),"unit":"eventos","quality":"Control interno","notes":"Correspondencias L↔R/R↔L."},
                            {"key":"v40_event_tolerance_ms","label":"Tolerancia temporal adaptativa V40","value":1000.0*float(pair_diag.get("v40_event_tolerance_s",np.nan)),"unit":"ms","quality":"Control interno","notes":"Umbral derivado de la resolución temporal real; nunca se interpreta como precisión sub-frame."},
                            {"key":"biplanar_overlap_duration_s","label":"Ventana biplanar sincronizada","value":float(end_s-start_s),"unit":"s","quality":sq,"notes":"Versión 35: intervalo simultáneo útil frontal/posterior + lateral tras corregir el desfase temporal."},
                            {"key":"biplanar_sync_coarse_offset_s","label":"Desfase global inicial frontal-lateral","value":float(pair_diag.get("coarse_offset_s",np.nan)),"unit":"s","quality":sq,"notes":"Estimación global basada en correlación del movimiento corporal."},
                            {"key":"biplanar_sync_fine_correction_s","label":"Ajuste continuo del reloj maestro V38","value":float(pair_diag.get("fine_correction_s",np.nan)),"unit":"s","quality":pair_diag.get("master_clock_quality","No calculable"),"notes":"Ajuste independiente de IC/TO usado como ancla física antes de la sincronización periódica V39."},
                            {"key":"biplanar_sync_final_offset_s","label":"Desfase temporal final frontal-lateral","value":float(pair_diag.get("final_offset_s",np.nan)),"unit":"s","quality":pair_q,"notes":"Versión 39: reloj V38 como ancla; puede incorporar refinamiento de fase sólo si ≥2 eventos laterales coherentes lo respaldan."},
                            {"key":"biplanar_master_clock_offset_s","label":"Reloj físico maestro · desfase lateral vs frontal","value":float(pair_diag.get("master_clock_offset_s",np.nan)),"unit":"s","quality":pair_diag.get("master_clock_quality","No calculable"),"notes":"Estimado con señal corporal continua de pelvis, tronco, rodillas y tobillos; independiente de IC (Initial Contact / Contacto Inicial)."},
                            {"key":"biplanar_master_clock_adjust_s","label":"Ajuste del reloj maestro respecto al desfase global","value":float(pair_diag.get("master_clock_adjust_s",np.nan)),"unit":"s","quality":pair_diag.get("master_clock_quality","No calculable"),"notes":"Corrección continua aplicada antes de proyectar eventos; los IC no modifican este valor."},
                            {"key":"biplanar_master_clock_correlation","label":"Correlación del reloj físico maestro","value":float(pair_diag.get("master_clock_correlation",np.nan)),"unit":"r","quality":pair_diag.get("master_clock_quality","No calculable"),"notes":"Correlación de la señal corporal común utilizada para fijar el reloj físico."},
                            {"key":"biplanar_front_step_period_s","label":"Periodo de paso frontal","value":float(pair_diag.get("front_step_period_s",np.nan)),"unit":"s","quality":"Control interno","notes":"Versión 39: mediana robusta entre IC alternantes L/R en la vista frontal/posterior."},
                            {"key":"biplanar_lateral_step_period_s","label":"Periodo de paso lateral","value":float(pair_diag.get("lateral_step_period_s",np.nan)),"unit":"s","quality":"Control interno","notes":"Versión 39: mediana robusta entre IC laterales aceptados alternantes L/R."},
                            {"key":"biplanar_step_period_difference_pct","label":"Diferencia relativa de periodo de paso entre cámaras","value":float(pair_diag.get("period_difference_pct",np.nan)),"unit":"%","quality":pair_diag.get("phase_quality","No calculable"),"notes":"Diferencia porcentual entre periodos frontal y lateral; ayuda a comprobar que ambas vistas observan la misma periodicidad."},
                            {"key":"biplanar_phase_sync_offset_s","label":"Desfase por fase periódica V39","value":float(pair_diag.get("phase_offset_s",np.nan)),"unit":"s","quality":pair_diag.get("phase_quality","No calculable"),"notes":"Offset propuesto por la secuencia periódica completa. Sólo se aplica si existen al menos dos eventos laterales coherentes."},
                            {"key":"biplanar_phase_sync_correction_s","label":"Corrección de fase respecto al reloj maestro","value":float(pair_diag.get("phase_correction_s",np.nan)),"unit":"s","quality":pair_diag.get("phase_quality","No calculable"),"notes":"Diferencia entre el offset periódico V39 y el reloj físico maestro V38."},
                            {"key":"biplanar_phase_sync_pairs","label":"Eventos que respaldan la sincronización de fase","value":float(pair_diag.get("phase_pairs",0)),"unit":"eventos","quality":pair_diag.get("phase_quality","No calculable"),"notes":"Número de eventos ordenados uno-a-uno que respaldan el ajuste periódico."},
                            {"key":"biplanar_phase_sync_mae_s","label":"MAE de sincronización periódica","value":float(pair_diag.get("phase_mae_s",np.nan)),"unit":"s","quality":pair_diag.get("phase_quality","No calculable"),"notes":"MAE (Mean Absolute Error / Error Absoluto Medio) entre eventos de fase emparejados."},
                            {"key":"biplanar_phase_sequence_error_s","label":"Error de coherencia entre intervalos de la secuencia","value":float(pair_diag.get("phase_sequence_error_s",np.nan)),"unit":"s","quality":pair_diag.get("phase_quality","No calculable"),"notes":"Diferencia media entre intervalos consecutivos frontal y lateral; independiente del offset absoluto."},
                            {"key":"biplanar_phase_supported","label":"Sincronización periódica respaldada por secuencia","value":float(pair_diag.get("phase_supported",0)),"unit":"bool","quality":pair_diag.get("phase_quality","No calculable"),"notes":"1 = V39 dispone de al menos dos eventos laterales coherentes y puede usar el offset de fase; 0 = conserva el reloj maestro V38."},
                            {"key":"biplanar_phase_correction_s","label":"Corrección de fase periódica de marcha","value":float(pair_diag.get("phase_correction_s",np.nan)),"unit":"s","quality":pair_diag.get("phase_quality","No calculable"),"notes":"Alias de compatibilidad V39: corrección periódica respecto al reloj físico maestro."},
                            {"key":"biplanar_phase_correlation","label":"Correlación de fase continua frontal-lateral (legado)","value":float(pair_diag.get("phase_correlation",np.nan)),"unit":"r","quality":"Control interno","notes":"Métrica heredada; V39 prioriza periodo de paso, secuencia y MAE de fase, no esta correlación."},
                            {"key":"biplanar_direct_pairs","label":"Pares válidos hipótesis L↔L / R↔R","value":float(pair_diag.get("direct_pairs",0)),"unit":"pares","quality":"Control interno","notes":"Correspondencia anatómica directa."},
                            {"key":"biplanar_swapped_pairs","label":"Pares válidos hipótesis L↔R / R↔L","value":float(pair_diag.get("swapped_pairs",0)),"unit":"pares","quality":"Control interno","notes":"Hipótesis de inversión izquierda/derecha entre cámaras."},
                            {"key":"biplanar_sequence_pairs","label":"Pares de secuencia biplanar consolidados","value":float(pair_diag.get("sequence_pairs",0)),"unit":"pares","quality":pair_q,"notes":"Versión 38: pares proyectados sobre el reloj físico maestro y aceptados considerando la secuencia completa, orden temporal, lado, IC y duración de ciclo."},
                            {"key":"biplanar_sequence_consistency_ms","label":"Consistencia temporal entre pares consecutivos","value":float(pair_diag.get("sequence_consistency_ms",np.nan)),"unit":"ms","quality":pair_q,"notes":"Dispersión del error temporal entre pares de la secuencia; menor es más coherente."},
                            {"key":"biplanar_second_lateral_homolog","label":"Segundo ciclo lateral con homólogo frontal · método de ciclos legado","value":float(pair_diag.get("second_lateral_homolog",0)),"unit":"bool","quality":pair_q,"notes":"1 = al menos dos ciclos laterales distintos tienen correspondencia frontal aceptada."},
                            {"key":"biplanar_single_pair_offset_guard","label":"Protección contra reajuste por un único par","value":float(pair_diag.get("single_pair_offset_guard",0)),"unit":"bool","quality":"Control interno","notes":"1 = se bloqueó un cambio grande del desfase porque estaba sustentado por un único par."},
                            {"key":"biplanar_master_clock_independent_ic","label":"Reloj maestro independiente de IC/TO","value":1.0,"unit":"bool","quality":"Control interno","notes":"Versión 39 conserva el reloj V38 como ancla independiente; la fase periódica sólo puede refinarlo con una secuencia de al menos dos eventos coherentes."},
                            {"key":"front_full_visible_cycles","label":"Ciclos detectados en ventana visible frontal/posterior","value":float(full_front_cycles),"unit":"ciclos","quality":"Control interno","notes":"Versión 35: ciclos detectados antes de recortar a la ventana biplanar común."},
                            {"key":"lateral_full_visible_cycles","label":"Ciclos detectados en ventana visible lateral","value":float(full_lateral_cycles),"unit":"ciclos","quality":(snap_lat.get("event_info") or {}).get("quality","Control interno") if snap_lat else "No calculable","notes":"Versión 35: ciclos laterales IC→IC aceptados por el detector multiseñal sobre toda la ventana visible lateral."},
                            {"key":"lateral_ic_candidates","label":"Candidatos IC (Initial Contact / Contacto Inicial) laterales","value":float((snap_lat.get("event_info") or {}).get("candidate_events",0)) if snap_lat else 0.0,"unit":"eventos","quality":"Control interno","notes":"Versión 35: candidatos detectados mediante posición relativa del pie, velocidad, trayectoria vertical y extensión de rodilla cuando está disponible."},
                            {"key":"lateral_ic_accepted","label":"IC (Initial Contact / Contacto Inicial) laterales aceptados","value":float((snap_lat.get("event_info") or {}).get("accepted_events",0)) if snap_lat else 0.0,"unit":"eventos","quality":(snap_lat.get("event_info") or {}).get("quality","No calculable") if snap_lat else "No calculable","notes":"Versión 35: eventos conservados tras filtros de plausibilidad temporal y alternancia izquierda-derecha."},
                            {"key":"biplanar_matched_cycles","label":"Ciclos biplanares coincidentes","value":float(len(pairs)),"unit":"ciclos","quality":pair_q,"notes":"Ciclos del mismo lado emparejados por IC (Initial Contact / Contacto Inicial) después de detectar cada ciclo en la ventana visible completa de su cámara."},
                            {"key":"biplanar_cycle_alignment_mae_s","label":"MAE (Mean Absolute Error / Error Absoluto Medio) temporal entre ciclos","value":pair_mae,"unit":"s","quality":pair_q,"notes":"Error temporal medio absoluto entre IC (Initial Contact / Contacto Inicial) frontal y lateral después de sincronización."},
                        ]
                        # La pestaña 9 usa frontal por defecto porque es la vista clínica activa.
                        st.session_state.cycle_seg = st.session_state.cycle_seg_front
                        st.session_state.cycle_fps = st.session_state.cycle_fps_front
                        st.session_state.cycle_support_left = st.session_state.cycle_support_left_front
                        st.session_state.cycle_support_right = st.session_state.cycle_support_right_front
                        st.session_state.cycle_source_view = "Frontal/posterior"
                        # Versión 34 · conservar solo datos ligeros para visor biplanar sincronizado.
                        st.session_state.biplanar_seg_front = seg_front.copy()
                        st.session_state.biplanar_seg_lateral = seg_lat.copy()
                        st.session_state.biplanar_fps_front = float(meta1["fps"])
                        st.session_state.biplanar_fps_lateral = float(meta2["fps"])
                        st.session_state.biplanar_sync_offset_s = float(pair_diag.get("final_offset_s",sync_user))
                        st.session_state.cycle_phase_review_base = None
                        st.session_state.cycle_phase_review_bytes = None
                        st.session_state.cycle_phase_review_speed = None

                        metrics = augment_metrics_with_gait_index(metrics, st.session_state.get("functional_support","Independiente"), st.session_state.get("measured_gait_speed",0.0), st.session_state.get("mode",""), st.session_state.get("view",""))
                        st.session_state.metrics = metrics
                        st.session_state.chart_front = chart_front
                        st.session_state.chart_lateral = chart_lat
                        st.session_state.metrics_done = True
                        st.markdown("#### Validación temporal · Versión 47 (congelada) + integración V51")
                        _v46_phys=float(pair_diag.get("v46_physical_clock_offset_s",np.nan))
                        _v46_phase=float(pair_diag.get("v46_biomechanical_phase_offset_s",np.nan))
                        _v46_delta=float(pair_diag.get("v46_phase_minus_physical_s",np.nan))
                        _v46_src=str(pair_diag.get("v46_physical_clock_source","No calculable"))
                        st.write(f"Reloj físico usado: **{_v46_phys:+.3f} s** ({_v46_src}) · fase biomecánica: **{_v46_phase:+.3f} s** · diferencia fase−reloj: **{1000*_v46_delta:+.1f} ms**")
                        if pair_diag.get("v46_ground_truth_active",0):
                            _v47_est=float(pair_diag.get("v47_independent_clock_estimate_s",np.nan))
                            _v47_err=1000.0*float(pair_diag.get("v47_clock_estimator_error_s",np.nan))
                            _v47_abs=1000.0*float(pair_diag.get("v47_clock_estimator_abs_error_s",np.nan))
                            _v47_fr=float(pair_diag.get("v47_clock_estimator_error_practical_frames",np.nan))
                            _v47_cls=str(pair_diag.get("v47_validation_class","No calculable"))
                            _v47_res=float(pair_diag.get("v47_practical_resolution_ms",np.nan))
                            st.success("Ground truth activo: gobierna el reloj y los ciclos; el estimador interno se evalúa de forma independiente.")
                            st.write(f"Estimación ciega V38: **{_v47_est:+.3f} s** · error firmado: **{_v47_err:+.1f} ms** · error absoluto: **{_v47_abs:.1f} ms** · **{_v47_fr:.2f} frames prácticos**.")
                            st.info(f"Clasificación V47: **{_v47_cls}** · resolución temporal práctica: **{_v47_res:.1f} ms**. Un valor sub-frame matemático no se interpreta como precisión física sub-frame.")
                        else:
                            st.caption("Modo clínico: no existe ground truth externo; V47 conserva el comportamiento de V46/V45 y no asigna un error de validación al estimador.")
                        st.session_state.ready_3d = ready
                        st.session_state.ready_3d_reasons = reasons
                        cloud_saved = False
                        if sb_ready() and st.session_state.get("cloud_session_id"):
                            try:
                                sb_save_metrics(st.session_state.cloud_session_id, metrics, start_s, end_s)
                                sb_update_session_3d(
                                    st.session_state.cloud_session_id,
                                    sync_offset_s=float(sync_user),
                                    sync_correlation=float(corr) if np.isfinite(corr) else None,
                                    sync_quality=sq,
                                    calibration_profile_name=st.session_state.get("selected_calibration_name"),
                                    ready_3d=bool(ready),
                                )
                                cloud_saved = True
                            except Exception as sb_e:
                                st.warning(f"Resultados calculados, pero no se pudieron guardar en Supabase: {sb_e}")
                        session_dir = Path(st.session_state.session_dir)

                        # v0.10.3 · copias limpias efímeras en memoria para auditoría
                        # del sujeto seleccionado en la pestaña 9.
                        try:
                            vp1 = Path(st.session_state.video1_path)
                            if vp1.exists():
                                st.session_state.cycle_video_clean_bytes = vp1.read_bytes()
                        except Exception:
                            st.session_state.cycle_video_clean_bytes = None
                        try:
                            vp2 = Path(st.session_state.video2_path)
                            if vp2.exists():
                                st.session_state.cycle_video2_clean_bytes = vp2.read_bytes()
                        except Exception:
                            st.session_state.cycle_video2_clean_bytes = None

                        try:
                            out1 = session_dir / "gait_front_web.mp4"
                            made1 = render_angle_video(Path(st.session_state.video1_path), df1, out1, "Frontal/posterior", st.session_state.get("assistive_device","Sin ayuda"))
                            if made1 and made1.exists():
                                st.session_state.annotated_video_bytes = made1.read_bytes()
                            out2 = session_dir / "gait_lateral_web.mp4"
                            made2 = render_angle_video(Path(st.session_state.video2_path), df2, out2, "Lateral", st.session_state.get("assistive_device","Sin ayuda"))
                            if made2 and made2.exists():
                                st.session_state.annotated_video2_bytes = made2.read_bytes()
                        except Exception:
                            st.session_state.annotated_video_bytes = None
                            st.session_state.annotated_video2_bytes = None
                        cleanup_temp_session(session_dir)
                        st.session_state.temp_deleted = True
                        st.session_state.analysis_df = None
                        st.session_state.analysis_df2 = None
                        st.success("✅ Resultados calculados y mostrados. ✅ Vídeos y archivos Pose2Sim eliminados del servidor temporal." + (" ✅ Histórico guardado en Supabase." if cloud_saved else " ⚠️ Histórico no guardado en Supabase."))
                    except Exception as e:
                        st.error(str(e))
                        with st.expander("Detalles técnicos"):
                            st.code(traceback.format_exc())
        else:
            fps = float(meta1["fps"])
            duration = (int(df1.frame.max()) + 1) / fps
            start_s, end_s = st.slider(
                "Intervalo válido (segundos)", 0.0, float(round(duration,2)),
                (0.0,float(round(duration,2))), step=max(0.01, round(1/fps,2))
            )
            if st.button("Calcular, guardar histórico y eliminar vídeo", type="primary", use_container_width=True):
                try:
                    start_frame = int(round(start_s*fps)); end_frame = int(round(end_s*fps))
                    metrics, chart, seg = compute_metrics(df1, fps, start_frame, end_frame, st.session_state.get("view",""), st.session_state.get("assistive_device","Sin ayuda"), st.session_state.get("scale_cm_per_px",0.0))
                    metrics += tracking_metrics(st.session_state.get("tracking_info_cam1"), "", st.session_state.get("view",""))

                    # v0.10.1 · preservar el segmento YA usado para los resultados
                    # antes de eliminar vídeo/JSON temporales.
                    snap = _make_cycle_snapshot(seg, fps)
                    st.session_state.cycle_seg = snap["seg"] if snap else seg.copy()
                    st.session_state.cycle_fps = float(fps)
                    st.session_state.cycle_support_left = snap["left"] if snap else None
                    st.session_state.cycle_support_right = snap["right"] if snap else None
                    st.session_state.cycle_source_view = st.session_state.get("view","")
                    st.session_state.cycle_phase_review_base = None
                    st.session_state.cycle_phase_review_bytes = None
                    st.session_state.cycle_phase_review_speed = None

                    metrics = augment_metrics_with_gait_index(metrics, st.session_state.get("functional_support","Independiente"), st.session_state.get("measured_gait_speed",0.0), st.session_state.get("mode",""), st.session_state.get("view",""))
                    st.session_state.metrics = metrics
                    st.session_state.chart = chart
                    st.session_state.metrics_done = True
                    cloud_saved = False
                    if sb_ready() and st.session_state.get("cloud_session_id"):
                        try:
                            sb_save_metrics(st.session_state.cloud_session_id, metrics, start_s, end_s)
                            cloud_saved = True
                        except Exception as sb_e:
                            st.warning(f"Resultados calculados, pero no se pudieron guardar en Supabase: {sb_e}")
                    session_dir = Path(st.session_state.session_dir)

                    # v0.10.3 · conservar en memoria SOLO durante la sesión una copia
                    # limpia del vídeo fuente. No se escribe en Supabase.
                    try:
                        vp_clean = Path(st.session_state.video1_path)
                        if vp_clean.exists():
                            st.session_state.cycle_video_clean_bytes = vp_clean.read_bytes()
                    except Exception:
                        st.session_state.cycle_video_clean_bytes = None

                    try:
                        out = session_dir / "gait_angles_web.mp4"
                        made = render_angle_video(Path(st.session_state.video1_path), df1, out, st.session_state.get("view",""), st.session_state.get("assistive_device","Sin ayuda"))
                        if made and made.exists():
                            st.session_state.annotated_video_bytes = made.read_bytes()
                    except Exception:
                        st.session_state.annotated_video_bytes = None
                    cleanup_temp_session(session_dir)
                    st.session_state.temp_deleted = True
                    st.session_state.analysis_df = None
                    st.success("✅ Resultados calculados y mostrados. ✅ Vídeo y archivos Pose2Sim eliminados del servidor temporal." + (" ✅ Histórico guardado en Supabase." if cloud_saved else " ⚠️ Histórico no guardado en Supabase."))
                except Exception as e:
                    st.error(str(e))
                    with st.expander("Detalles técnicos"):
                        st.code(traceback.format_exc())

    if st.session_state.get("metrics_done"):
        metrics = st.session_state.metrics
        st.markdown("### Resumen")
        if st.session_state.get("mode", "").startswith("2 cámaras"):
            c1,c2,c3,c4 = st.columns(4)
            c1.metric("Cadencia lateral", fmt(metric_value(metrics,"lateral_cadence_exp"),1)+" pasos/min")
            c2.metric("Tracking frontal", fmt(metric_value(metrics,"front_good_frames_pct"),1)+" %")
            c3.metric("Tracking lateral", fmt(metric_value(metrics,"lateral_good_frames_pct"),1)+" %")
            c4.metric("Desfase cámara lateral", fmt(metric_value(metrics,"sync_offset_cam02_s"),2)+" s")
            st.markdown("#### Correspondencia biplanar de ciclos")
            b1,b2,b3=st.columns(3)
            b1.metric("Ventana simultánea útil", fmt(metric_value(metrics,"biplanar_overlap_duration_s"),2)+" s")
            b2.metric("Ciclos coincidentes", fmt(metric_value(metrics,"biplanar_matched_cycles"),0))
            b3.metric("MAE (Mean Absolute Error / Error Absoluto Medio) temporal", fmt(metric_value(metrics,"biplanar_cycle_alignment_mae_s"),3)+" s")
            st.markdown("##### Correspondencia temporal frontal-lateral · Versión 45")
            s1,s2,s3=st.columns(3)
            s1.metric("Desfase global inicial", fmt(metric_value(metrics,"biplanar_sync_coarse_offset_s"),3)+" s")
            s2.metric("Ajuste del reloj maestro", fmt(metric_value(metrics,"biplanar_master_clock_adjust_s"),3)+" s")
            s3.metric("Reloj físico maestro · desfase final", fmt(metric_value(metrics,"biplanar_master_clock_offset_s"),3)+" s")
            _mc=metric_value(metrics,"biplanar_master_clock_correlation")
            _mcq="No calculable"
            if np.isfinite(_mc):
                _mcq="Alta" if _mc>=0.65 else ("Moderada" if _mc>=0.40 else "Baja")
            st.caption(f"Correlación del reloj físico maestro: {fmt(_mc,3)} · calidad {_mcq}. Los IC (Initial Contact / Contacto Inicial) se proyectan después y no reajustan el reloj.")
            h1,h2,h3=st.columns(3)
            h1.metric("Pares · hipótesis directa L↔L / R↔R", fmt(metric_value(metrics,"biplanar_direct_pairs"),0))
            h2.metric("Pares · hipótesis invertida L↔R / R↔L", fmt(metric_value(metrics,"biplanar_swapped_pairs"),0))
            _hyp="No calculable"
            if metric_value(metrics,"biplanar_swapped_pairs")>metric_value(metrics,"biplanar_direct_pairs"):
                _hyp="L↔R / R↔L invertida"
            elif metric_value(metrics,"biplanar_direct_pairs")>0:
                _hyp="L↔L / R↔R directa"
            h3.markdown("**Hipótesis seleccionada**")
            h3.markdown(_hyp)
            p1,p2,p3=st.columns(3)
            p1.metric("Periodo de paso frontal", fmt(metric_value(metrics,"biplanar_front_step_period_s"),3)+" s")
            p2.metric("Periodo de paso lateral", fmt(metric_value(metrics,"biplanar_lateral_step_period_s"),3)+" s")
            p3.metric("Diferencia de periodo", fmt(metric_value(metrics,"biplanar_step_period_difference_pct"),1)+" %")
            q1,q2,q3=st.columns(3)
            q1.metric("Corrección de fase V39", fmt(metric_value(metrics,"biplanar_phase_sync_correction_s"),3)+" s")
            q2.metric("Eventos que respaldan la fase", fmt(metric_value(metrics,"biplanar_phase_sync_pairs"),0))
            _phase_ok=metric_value(metrics,"biplanar_phase_supported")
            q3.metric("Fase periódica aplicada", "Sí" if np.isfinite(_phase_ok) and _phase_ok>=0.5 else "No")
            _pmae=metric_value(metrics,"biplanar_phase_sync_mae_s")
            _pseq=metric_value(metrics,"biplanar_phase_sequence_error_s")
            st.caption(
                f"Sincronización periódica V39: MAE (Mean Absolute Error / Error Absoluto Medio)={fmt(_pmae,3)} s; "
                f"error de coherencia entre intervalos={fmt(_pseq,3)} s. "
                "El reloj V38 se mantiene como ancla y sólo se refina con ≥2 eventos laterales coherentes."
            )
            st.markdown("##### Evaluabilidad del segundo ciclo · Versión 45")
            _v45status=str(st.session_state.get("biplanar_v45_status","No evaluable") or "No evaluable")
            _v45reason=str(st.session_state.get("biplanar_v45_reason","") or "")
            _v45last_side=str(st.session_state.get("biplanar_v45_last_accepted_side","") or "")
            _v45last_frame=st.session_state.get("biplanar_v45_last_accepted_frame",np.nan)
            _v45next_side=str(st.session_state.get("biplanar_v45_expected_next_side","") or "")
            _v45eval=metric_value(metrics,"v45_second_cycle_evaluable")
            _v45cov=metric_value(metrics,"v45_coverage_sufficient")
            _v45pred=metric_value(metrics,"v45_predicted_next_ic_frame")
            _v45end=metric_value(metrics,"v45_visibility_end_frame")
            _v45margin=metric_value(metrics,"v45_visibility_margin_ms")
            v451,v452,v453=st.columns(3)
            v451.metric("Segundo ciclo evaluable","Sí" if np.isfinite(_v45eval) and _v45eval>=0.5 else "No")
            v452.metric("Cobertura lateral suficiente","Sí" if np.isfinite(_v45cov) and _v45cov>=0.5 else "No")
            v453.metric("Margen tras IC esperado",f"{_v45margin:.1f} ms" if np.isfinite(_v45margin) else "No calculable")
            st.info(f"**Estado V45:** {_v45status}. {_v45reason}")
            if np.isfinite(_v45pred) or np.isfinite(_v45end):
                _v45df=pd.DataFrame([{
                    "Último IC aceptado":f"{_v45last_side} · frame {_v45last_frame}",
                    "Siguiente lado esperado":_v45next_side,
                    "Frame esperado siguiente IC":round(float(_v45pred),2) if np.isfinite(_v45pred) else None,
                    "Último frame visible lateral":round(float(_v45end),2) if np.isfinite(_v45end) else None,
                    "Margen de cobertura (ms)":round(float(_v45margin),1) if np.isfinite(_v45margin) else None,
                }])
                st.dataframe(_v45df,width="stretch",hide_index=True)
            _near45=st.session_state.get("biplanar_v45_near_prediction_candidates") or []
            if _near45:
                with st.expander("Candidatos laterales próximos al siguiente IC esperado"):
                    st.dataframe(pd.DataFrame(_near45),width="stretch",hide_index=True)
            st.caption(
                "V45 evita interpretar la ausencia del segundo ciclo como evidencia negativa cuando la cámara lateral deja de cubrir al paciente antes del siguiente IC esperado. "
                "No crea eventos, no rescata contactos y no modifica el reloj biplanar."
            )

            st.markdown("##### Encadenamiento de ciclos consecutivos · Versión 44")
            _v44chains=st.session_state.get("biplanar_v44_chains") or []
            _v44audit=st.session_state.get("biplanar_v44_chain_audit") or []
            v441,v442,v443=st.columns(3)
            v441.metric("Cadenas consecutivas",fmt(metric_value(metrics,"v44_chain_count"),0))
            v442.metric("Ciclos homólogos encadenados",fmt(metric_value(metrics,"v44_chained_cycles"),0))
            _v44second=metric_value(metrics,"v44_second_cycle_homolog")
            v443.metric("Segundo ciclo lateral homólogo","Sí" if np.isfinite(_v44second) and _v44second>=0.5 else "No")
            if _v44chains:
                _rows44=[]
                for _i,_x in enumerate(_v44chains,1):
                    _d=_x.get("delta_events_s",[])
                    _rows44.append({
                        "Cadena":_i,
                        "Hipótesis":_x.get("hypothesis",""),
                        "Frames frontal":"-".join(map(str,_x.get("front_frames",[]))),
                        "Frames lateral":"-".join(map(str,_x.get("lateral_frames",[]))),
                        "IC compartido frontal":_x.get("shared_front_frame"),
                        "IC compartido lateral":_x.get("shared_lateral_frame"),
                        "Ciclo 1 frontal (ms)":round(1000*float(_x.get("cycle1_front_s",np.nan)),1),
                        "Ciclo 1 lateral (ms)":round(1000*float(_x.get("cycle1_lateral_s",np.nan)),1),
                        "Δ ciclo 1 (ms)":round(1000*float(_x.get("cycle1_duration_delta_s",np.nan)),1),
                        "Ciclo 2 frontal (ms)":round(1000*float(_x.get("cycle2_front_s",np.nan)),1),
                        "Ciclo 2 lateral (ms)":round(1000*float(_x.get("cycle2_lateral_s",np.nan)),1),
                        "Δ ciclo 2 (ms)":round(1000*float(_x.get("cycle2_duration_delta_s",np.nan)),1),
                        "Δ IC1..IC5 (ms)":" / ".join(f"{1000*abs(float(q)):.1f}" for q in _d),
                        "Discrepancia IC medio ciclo 1":bool(_x.get("middle1_discrepancy")),
                        "Discrepancia IC medio ciclo 2":bool(_x.get("middle2_discrepancy")),
                    })
                st.dataframe(pd.DataFrame(_rows44),use_container_width=True,hide_index=True)
            elif _v44audit:
                _dfa44=pd.DataFrame(_v44audit)
                st.warning("V44 no ha validado dos ciclos consecutivos. La tabla de auditoría muestra qué criterio bloquea la cadena.")
                st.dataframe(_dfa44,use_container_width=True,hide_index=True)
            else:
                st.info("No hay cinco IC (Initial Contact / Contacto Inicial) alternantes suficientes en ambas vistas para construir dos ciclos consecutivos.")
            st.caption(
                "V44 exige continuidad temporal estructural: IC1→IC2→IC3 forma el primer ciclo e IC3→IC4→IC5 el segundo. "
                "El mismo IC3 debe cerrar el ciclo 1 y abrir el ciclo 2 en ambas cámaras. No se crean eventos, no se reajusta el reloj y no se permite saltar eventos dentro de la cadena."
            )

            st.markdown("##### Validación por cierre de ciclo · Versión 43")
            _v43cycles=st.session_state.get("biplanar_v43_closure_cycles") or []
            _v43n=metric_value(metrics,"v43_closure_cycles")
            _v43hom=metric_value(metrics,"v43_second_lateral_homolog")
            c431,c432,c433=st.columns(3)
            c431.metric("Ciclos homólogos por cierre", fmt(_v43n,0))
            c432.metric("Segundo ciclo lateral homólogo", "Sí" if np.isfinite(_v43hom) and _v43hom>=0.5 else "No")
            _mid43=sum(1 for _x in _v43cycles if _x.get("middle_ic_discrepancy"))
            c433.metric("Con discrepancia IC intermedio", _mid43)
            if _v43cycles:
                _rows43=[]
                _lfps43=metric_value(metrics,"v40_lateral_fps")
                for _i,_x in enumerate(_v43cycles,1):
                    _rows43.append({
                        "Ciclo":_i,
                        "Clasificación":_x.get("classification",""),
                        "Lado frontal":_x.get("side",""),
                        "Frames frontal":f'{_x.get("front_start_frame")}-{_x.get("front_mid_frame")}-{_x.get("front_end_frame")}',
                        "Frames lateral":f'{_x.get("lateral_start_frame")}-{_x.get("lateral_mid_frame")}-{_x.get("lateral_end_frame")}',
                        "Duración frontal (ms)":round(1000*float(_x.get("front_duration_s",np.nan)),1),
                        "Duración lateral (ms)":round(1000*float(_x.get("lateral_duration_s",np.nan)),1),
                        "Δ duración (ms)":round(1000*float(_x.get("duration_delta_s",np.nan)),1),
                        "Δ IC inicial (ms)":round(1000*abs(float(_x.get("start_delta_s",np.nan))),1),
                        "Δ IC intermedio (ms)":round(1000*abs(float(_x.get("mid_delta_s",np.nan))),1),
                        "Δ IC final (ms)":round(1000*abs(float(_x.get("end_delta_s",np.nan))),1),
                        "Δ IC medio (frames lateral)":round(abs(float(_x.get("mid_delta_s",np.nan)))*_lfps43,2) if np.isfinite(_lfps43) else None,
                    })
                st.dataframe(pd.DataFrame(_rows43),use_container_width=True,hide_index=True)
            st.caption(
                "V43 no corrige el IC (Initial Contact / Contacto Inicial) intermedio ni amplía la sincronización. "
                "Acepta un ciclo como homólogo cuando los extremos del ciclo, la alternancia, la duración total y la periodicidad lateral son coherentes. "
                "Si el IC frontal intermedio está desplazado, el ciclo se conserva pero queda etiquetado explícitamente como discrepancia localizada."
            )

            st.markdown("##### Auditoría exhaustiva del triplete candidato · Versión 42")
            _a42=st.session_state.get("biplanar_v42_triplet_audit") or []
            _c42=st.session_state.get("biplanar_v42_candidate_audit") or []
            if _a42:
                _df42=pd.DataFrame(_a42)
                _accepted42=int(_df42["accepted"].sum()) if "accepted" in _df42 else 0
                a1,a2,a3=st.columns(3)
                a1.metric("Tripletes auditados", len(_df42))
                a2.metric("Tripletes aceptados", _accepted42)
                a3.metric("Tripletes rechazados", len(_df42)-_accepted42)
                st.dataframe(_df42,use_container_width=True,hide_index=True)
                _rej42=_df42.loc[~_df42["accepted"],"rejection_reason"].tolist() if "accepted" in _df42 else []
                if _rej42:
                    st.warning("Motivo exacto de rechazo V42: "+" || ".join(map(str,_rej42)))
            else:
                st.info("V42 no dispuso de tres correspondencias ordenadas para auditar un triplete completo.")
            if _c42:
                with st.expander("Auditoría de candidatos periódicos V42"):
                    st.dataframe(pd.DataFrame(_c42),use_container_width=True,hide_index=True)
            st.caption(
                "V42 es diagnóstica: no relaja umbrales. Muestra frames, tiempos, intervalos, errores respecto al periodo esperado, "
                "alternancia L/R, diferencia de duración y el criterio exacto que acepta o rechaza cada triplete."
            )

            st.markdown("##### Validación periódica del tercer contacto · Versión 41")
            q1,q2,q3,q4=st.columns(4)
            q1.metric("IC recuperados", fmt(metric_value(metrics,"v41_rescued_events"),0))
            q2.metric("Ciclos V41 validados", fmt(metric_value(metrics,"v41_triplet_cycles"),0))
            _v41hom=metric_value(metrics,"v41_second_lateral_homolog")
            q3.metric("Segundo ciclo lateral homólogo", "Sí" if np.isfinite(_v41hom) and _v41hom>=0.5 else "No")
            q4.metric("Ventana local de rescate", fmt(metric_value(metrics,"v41_rescue_tolerance_ms"),1)+" ms")
            st.caption(
                "V41 no amplía indiscriminadamente la tolerancia ni crea IC (Initial Contact / Contacto Inicial). "
                "Solo recupera un evento realmente detectado si aparece cerca del instante previsto por el periodo de paso, "
                "respeta la alternancia L/R y mantiene coherencia temporal entre ambas cámaras."
            )
            _v41rows=st.session_state.get("biplanar_v41_rescue_rows") or []
            if _v41rows:
                _df41=pd.DataFrame(_v41rows).copy()
                _df41["Δ frontal-predicción (ms)"]=1000.0*_df41["prediction_error_front_s"]
                _df41["Δ lateral-predicción (ms)"]=1000.0*_df41["prediction_error_lateral_s"]
                _df41["Δ biplanar (ms)"]=1000.0*_df41["cross_delta_s"].abs()
                st.dataframe(_df41[["front_side","front_frame","lateral_side","lateral_frame",
                                    "Δ frontal-predicción (ms)","Δ lateral-predicción (ms)","Δ biplanar (ms)"]],
                             use_container_width=True,hide_index=True)

            st.markdown("##### Reconstrucción biplanar por secuencia de pasos · Versión 40")
            f1,f2,f3,f4=st.columns(4)
            f1.metric("FPS frontal", fmt(metric_value(metrics,"v40_front_fps"),1))
            f2.metric("FPS lateral", fmt(metric_value(metrics,"v40_lateral_fps"),1))
            f3.metric("1 frame lateral", fmt(metric_value(metrics,"v40_lateral_frame_ms"),1)+" ms")
            f4.metric("Incertidumbre temporal", "±"+fmt(metric_value(metrics,"v40_temporal_uncertainty_ms"),1)+" ms")
            w1,w2,w3=st.columns(3)
            w1.metric("Pares de pasos sincronizados", fmt(metric_value(metrics,"v40_event_pairs"),0))
            w2.metric("Ciclos por triplete IC→contra→IC", fmt(metric_value(metrics,"v40_triplet_cycles"),0))
            _v40hom=metric_value(metrics,"v40_second_lateral_homolog")
            w3.metric("Segundo ciclo lateral con homólogo", "Sí" if np.isfinite(_v40hom) and _v40hom>=0.5 else "No")
            st.caption(
                f"MAE (Mean Absolute Error / Error Absoluto Medio) de pasos: {fmt(metric_value(metrics,'v40_event_mae_ms'),1)} ms; "
                f"error de secuencia: {fmt(metric_value(metrics,'v40_event_sequence_error_ms'),1)} ms. "
                "Las diferencias temporales se interpretan también en frames y no se atribuye precisión por debajo de la resolución de la cámara más lenta."
            )
            _v40tr=st.session_state.get("biplanar_v40_triplets") or []
            if _v40tr:
                _v40rows=[]
                _ff40=metric_value(metrics,"v40_front_fps")
                _lf40=metric_value(metrics,"v40_lateral_fps")
                for _i,_t in enumerate(_v40tr,1):
                    _v40rows.append({
                        "Ciclo":_i,
                        "Lado frontal":"Izquierdo" if _t.get("side")=="L" else "Derecho",
                        "Hipótesis":"L/R invertida" if _t.get("lr_swapped") else "L/R directa",
                        "Duración frontal (s)":round(float(_t.get("front_duration_s",np.nan)),3),
                        "Duración lateral (s)":round(float(_t.get("lateral_duration_s",np.nan)),3),
                        "Δ duración (ms)":round(1000.0*float(_t.get("duration_delta_s",np.nan)),1),
                        "Δ IC inicial (ms)":round(1000.0*abs(float(_t.get("start_delta_s",np.nan))),1),
                        "Δ IC contralateral (ms)":round(1000.0*abs(float(_t.get("mid_delta_s",np.nan))),1),
                        "Δ IC final (ms)":round(1000.0*abs(float(_t.get("end_delta_s",np.nan))),1),
                        "Δ inicial (frames lateral)":round(abs(float(_t.get("start_delta_s",np.nan)))*_lf40,2) if np.isfinite(_lf40) else None,
                        "Δ medio (frames lateral)":round(abs(float(_t.get("mid_delta_s",np.nan)))*_lf40,2) if np.isfinite(_lf40) else None,
                        "Δ final (frames lateral)":round(abs(float(_t.get("end_delta_s",np.nan)))*_lf40,2) if np.isfinite(_lf40) else None,
                    })
                st.dataframe(pd.DataFrame(_v40rows),use_container_width=True,hide_index=True)

            v1,v2,v3=st.columns(3)
            v1.metric("Pares consolidados de secuencia · método previo", fmt(metric_value(metrics,"biplanar_sequence_pairs"),0))
            _hom=metric_value(metrics,"biplanar_second_lateral_homolog")
            v2.metric("Segundo homólogo · método previo", "Sí" if np.isfinite(_hom) and _hom>=0.5 else "No")
            _phase_ok2=metric_value(metrics,"biplanar_phase_supported")
            v3.metric("Refinamiento periódico", "Respaldado" if np.isfinite(_phase_ok2) and _phase_ok2>=0.5 else "No aplicado")
            _sc=metric_value(metrics,"biplanar_sequence_consistency_ms")
            if np.isfinite(_sc):
                st.caption(f"Consistencia temporal de la secuencia: {_sc:.1f} ms de dispersión entre errores de emparejamiento.")

            st.markdown("##### Línea temporal diagnóstica de IC (Initial Contact / Contacto Inicial)")
            _fs=st.session_state.get("cycle_support_left_front")
            _fr=st.session_state.get("cycle_support_right_front")
            _ls=st.session_state.get("cycle_support_left_lateral")
            _lr=st.session_state.get("cycle_support_right_lateral")
            _ff=float(st.session_state.get("cycle_fps_front") or 0)
            _lf=float(st.session_state.get("cycle_fps_lateral") or 0)
            _off=metric_value(metrics,"biplanar_sync_final_offset_s")
            _events=[]
            if _ff>0:
                for _side,_obj in [("I",_fs),("D",_fr)]:
                    if _obj:
                        _xs=list(np.asarray(_obj.get("ic",[]),dtype=float))+list(np.asarray(_obj.get("next_ic",[]),dtype=float))
                        for _x in sorted(set(float(v) for v in _xs if np.isfinite(v))):
                            _events.append({"Vista":"Frontal","Lado":_side,"Tiempo físico (s)":round(float(_x/_ff),3)})
            if _lf>0 and np.isfinite(_off):
                for _side,_obj in [("I",_ls),("D",_lr)]:
                    if _obj:
                        _xs=list(np.asarray(_obj.get("ic",[]),dtype=float))+list(np.asarray(_obj.get("next_ic",[]),dtype=float))
                        for _x in sorted(set(float(v) for v in _xs if np.isfinite(v))):
                            _events.append({"Vista":"Lateral alineada","Lado":_side,"Tiempo físico (s)":round(float(_x/_lf-_off),3)})
            if _events:
                st.dataframe(pd.DataFrame(_events).sort_values("Tiempo físico (s)"),use_container_width=True,hide_index=True)
            _pairs=st.session_state.get("biplanar_cycle_pairs") or []
            if _pairs:
                _rows=[]
                for _i,_p in enumerate(_pairs,1):
                    _rows.append({
                        "Ciclo":_i,
                        "Lado": "Izquierdo" if _p.get("side")=="L" else "Derecho",
                        "IC frontal (s)":round(float(_p["front"]["ic_s"]),3),
                        "IC lateral alineado (s)":round(float(_p["lateral"]["ic_s"]-metric_value(metrics,"biplanar_sync_final_offset_s")),3),
                        "Diferencia IC (ms)":round(1000.0*float(_p.get("delta_s",0.0)),1),
                        "Diferencia duración ciclo (ms)":round(1000.0*float(_p.get("duration_delta_s",0.0)),1),
                    })
                st.dataframe(pd.DataFrame(_rows),use_container_width=True,hide_index=True)
            else:
                st.info("No se han aceptado pares de ciclos frontal-lateral. Sentinel no genera correspondencias biplanares si los IC (Initial Contact / Contacto Inicial) no cumplen los criterios temporales y de lado.")
            st.markdown("##### Diagnóstico del detector lateral")
            d1,d2,d3=st.columns(3)
            d1.metric("Candidatos IC laterales", fmt(metric_value(metrics,"lateral_ic_candidates"),0))
            d2.metric("IC laterales aceptados", fmt(metric_value(metrics,"lateral_ic_accepted"),0))
            d3.metric("Ciclos IC→IC laterales", fmt(metric_value(metrics,"lateral_full_visible_cycles"),0))
            st.caption("IC (Initial Contact / Contacto Inicial). La Versión 34 separa candidatos de eventos aceptados para auditar el detector sagital antes del emparejamiento frontal-lateral.")
            _lat_info=st.session_state.get("biplanar_lateral_event_info") or {}
            if _lat_info and _lat_info.get("candidate_audit"):
                _audit_rows=[]
                _off_a=metric_value(metrics,"biplanar_sync_final_offset_s")
                _lf_a=float(st.session_state.get("cycle_fps_lateral") or 0)
                for _i,_ev in enumerate(_lat_info.get("candidate_audit",[]),1):
                    _fr=_ev.get("frame",np.nan)
                    _raw=float(_fr/_lf_a) if _lf_a>0 and np.isfinite(_fr) else np.nan
                    _phys=float(_raw-_off_a) if np.isfinite(_raw) and np.isfinite(_off_a) else np.nan
                    _audit_rows.append({
                        "Nº":_i,
                        "Lado":_ev.get("side","?"),
                        "Frame":int(_fr) if np.isfinite(_fr) else None,
                        "Tiempo cámara (s)":round(_raw,3) if np.isfinite(_raw) else None,
                        "Tiempo físico V39 (s)":round(_phys,3) if np.isfinite(_phys) else None,
                        "Score":round(float(_ev.get("score",np.nan)),3) if np.isfinite(_ev.get("score",np.nan)) else None,
                        "Estado":_ev.get("status",""),
                        "Motivo":_ev.get("reason",""),
                    })
                st.markdown("###### Auditoría completa de candidatos IC laterales · V39")
                st.dataframe(pd.DataFrame(_audit_rows),use_container_width=True,hide_index=True)
            pq=st.session_state.get("biplanar_cycle_quality","No calculable")
            if pq=="Alta": st.success("Correspondencia de ciclos frontal-lateral: Alta.")
            elif pq=="Moderada": st.info("Correspondencia de ciclos frontal-lateral: Moderada.")
            else: st.warning("Correspondencia de ciclos frontal-lateral: Baja o no calculable. No usar esta sesión como base para cinética estimada sin revisión.")
            st.caption("IC (Initial Contact / Contacto Inicial) se usa como referencia temporal para emparejar el mismo ciclo de cada lado en ambas cámaras. 2D (Two-Dimensional / Bidimensional).")

            st.markdown("#### Cinemática biplanar sincronizada")
            st.caption("Versión 34: ambas cámaras se remuestrean sobre un reloj común en segundos. La vista frontal aporta control mediolateral y la lateral aporta cinemática sagital. No es todavía una reconstrucción 3D (Three-Dimensional / Tridimensional).")
            bf=st.session_state.get("biplanar_seg_front")
            bl=st.session_state.get("biplanar_seg_lateral")
            if isinstance(bf,pd.DataFrame) and not bf.empty and isinstance(bl,pd.DataFrame) and not bl.empty:
                try:
                    kf=_compute_display_kinematics(bf); kl=_compute_display_kinematics(bl)
                    ff=float(st.session_state.get("biplanar_fps_front",30.0)); fl=float(st.session_state.get("biplanar_fps_lateral",30.0))
                    tf=(pd.to_numeric(kf["frame"],errors="coerce").to_numpy(float)-pd.to_numeric(kf["frame"],errors="coerce").min())/ff
                    tl=(pd.to_numeric(kl["frame"],errors="coerce").to_numpy(float)-pd.to_numeric(kl["frame"],errors="coerce").min())/fl
                    front_vars=[x for x in ["Oblicuidad pélvica (°)","Inclinación tronco (°)","Oblicuidad hombros (°)"] if x in kf.columns]
                    lat_vars=[x for x in ["Flexión cadera Izquierda (°)","Flexión cadera Derecha (°)","Flexión rodilla Izquierda (°)","Flexión rodilla Derecha (°)","Ángulo tobillo Izquierda (°)","Ángulo tobillo Derecha (°)"] if x in kl.columns]
                    qf,ql=st.columns(2)
                    fv=qf.selectbox("Variable frontal sincronizada",front_vars,key="v32_biplanar_front_var") if front_vars else None
                    lv=ql.selectbox("Variable lateral sincronizada",lat_vars,index=min(2,len(lat_vars)-1),key="v32_biplanar_lat_var") if lat_vars else None
                    fig,ax=plt.subplots(figsize=(10,3.4))
                    plotted=False
                    if fv:
                        y=pd.to_numeric(kf[fv],errors="coerce").to_numpy(float); ok=np.isfinite(tf)&np.isfinite(y)
                        if ok.sum()>2: ax.plot(tf[ok],y[ok],label=f"Frontal · {fv}"); plotted=True
                    if lv:
                        y=pd.to_numeric(kl[lv],errors="coerce").to_numpy(float); ok=np.isfinite(tl)&np.isfinite(y)
                        if ok.sum()>2: ax.plot(tl[ok],y[ok],label=f"Lateral · {lv}"); plotted=True
                    ax.set_xlabel("Tiempo físico compartido (s)"); ax.set_ylabel("Ángulo proyectado 2D (°)"); ax.set_title("Cinemática frontal + lateral sobre el mismo reloj temporal")
                    ax.grid(True,alpha=.2)
                    if plotted: ax.legend(fontsize=8)
                    st.pyplot(fig,use_container_width=True); plt.close(fig)
                    if not pairs:
                        st.warning("Hay cinemática temporal sincronizada, pero no hay ciclos IC→IC válidos simultáneos. Por seguridad no se genera todavía una curva biplanar normalizada 0–100 % del ciclo.")
                    else:
                        st.success(f"{len(pairs)} ciclo(s) IC→IC biplanar(es) validado(s). La normalización 0–100 % puede usarse como base para la siguiente etapa.")
                except Exception as e:
                    st.warning(f"No se pudo construir el visor cinemático biplanar: {e}")
            else:
                st.info("Calcula primero ambas vistas para construir la cinemática biplanar sincronizada.")

            st.caption(f"Cadencia lateral ({reference_text_for_metric('lateral_cadence_exp')})")
            sl=metric_value(metrics,"lateral_stance_time_l_2d"); sr=metric_value(metrics,"lateral_stance_time_r_2d"); sa=metric_value(metrics,"lateral_stance_asymmetry_2d")
            qsl=metric_value(metrics,"lateral_support_segmentation_score_l"); qsr=metric_value(metrics,"lateral_support_segmentation_score_r")
            if sl is not None and sr is not None and np.isfinite(sl) and np.isfinite(sr):
                side="IZQUIERDA" if sl>sr else ("DERECHA" if sr>sl else "SIMILAR")
                st.info(f"Apoyo 2D por ciclos continuos (lateral): I {sl:.2f} s · D {sr:.2f} s · asimetría {fmt(sa,1)} % · mayor tiempo estimado: **{side}**. ({reference_text_for_metric('lateral_stance_asymmetry_2d')})")
            else:
                st.warning(f"Tiempo de apoyo lateral NO informado: la segmentación apoyo/oscilación no superó el control temporal (I {fmt(qsl,0)}/100 · D {fmt(qsr,0)}/100).")
            st.markdown("#### Consistencia temporal · cámara lateral")
            t1,t2,t3,t4 = st.columns(4)
            t1.metric("Eventos detectados", fmt(metric_value(metrics,"lateral_step_events_detected"),0))
            t2.metric("Duración segmento", fmt(metric_value(metrics,"lateral_segment_duration_s"),2)+" s")
            t3.metric("Eventos esperados", fmt(metric_value(metrics,"lateral_expected_steps_from_cadence"),1))
            t4.metric("Discrepancia", fmt(metric_value(metrics,"lateral_step_count_consistency_error_pct"),1)+" %")
            cq = metric_quality(metrics,"lateral_step_count_consistency_error_pct") or "No calculable"
            if cq == "Alta": st.success(f"Consistencia interna: {cq}")
            elif cq == "Moderada": st.info(f"Consistencia interna: {cq}")
            else: st.warning(f"Consistencia interna: {cq}. Conviene revisar visualmente los eventos detectados.")
            st.caption("Control interno: número de eventos de alternancia detectados ↔ cadencia estimada ↔ duración. Los eventos siguen siendo experimentales y no equivalen todavía a heel-strikes validados.")
            st.markdown("### Resumen biomecánico")
            for paragraph in biomech_summary(metrics, "Lateral", prefix="lateral_"):
                st.write(paragraph)
            for paragraph in biomech_summary(metrics, "Frontal/posterior", prefix="front_")[1:]:
                st.write(paragraph)
            st.markdown("### Regularidad y consistencia paso a paso · cámara lateral")
            _r1,_r2,_r3=st.columns(3)
            _r1.metric("Consistencia paso a paso", fmt(metric_value(metrics,"lateral_step_to_step_similarity_index"),1)+" /100")
            _r2.metric("Cambio entre pasos", fmt(metric_value(metrics,"lateral_step_to_step_change_pct"),1)+" %")
            _r3.metric("CV temporal", fmt(metric_value(metrics,"lateral_regularity_cv"),1)+" %")
            st.caption("La consistencia compara cada paso con el inmediatamente anterior; la simetría compara izquierda con derecha.")
            ready = bool(st.session_state.get("ready_3d", False))
            if ready:
                st.success("🧭 Sesión preparada para una futura triangulación 3D: dos poses, sincronización aceptable, visibilidad suficiente y calibración seleccionada.")
            else:
                st.warning("La sesión todavía no cumple todos los requisitos de preparación 3D.")
                for reason in st.session_state.get("ready_3d_reasons", []):
                    st.write(f"• {reason}")
            st.warning("La Versión 54 **no triangula todavía coordenadas 3D (Three-Dimensional / Tridimensional)**. Los resultados siguientes continúan siendo 2D, pero cada plano se interpreta desde la cámara apropiada.")
            chart_front = st.session_state.get("chart_front")
            chart_lat = st.session_state.get("chart_lateral")
            if chart_front is not None:
                st.markdown("### Cámara 1 · frontal/posterior")
                st.markdown("#### Centro de masa estimado (COM (Centro de Masa estimado 2D)) y estabilidad dinámica")
                _c1,_c2,_c3,_c4=st.columns(4)
                _c1.metric("Excursión ML", fmt(metric_value(metrics,"front_com_lateral_excursion_norm_pct"),1)+" % pelvis")
                _c2.metric("Excursión imagen", fmt(metric_value(metrics,"front_com_lateral_excursion_px"),1)+" px")
                _c3.metric("Velocidad ML P95", fmt(metric_value(metrics,"front_com_velocity_ml_px_s"),1)+" px/s")
                _c4.metric("COM (Centro de Masa estimado 2D) / BOS (Base de Sustentación) P95", fmt(metric_value(metrics,"front_com_bos_ratio_p95"),2))
                _ccols=[c for c in ["COM normalizado (% pelvis)","COM respecto BOS (ratio)"] if c in chart_front.columns]
                if _ccols:
                    st.line_chart(chart_front.set_index("time_s")[_ccols], x_label="Tiempo (s)", y_label="COM (Centro de Masa estimado 2D) normalizado")
                st.caption("COM (Centro de Masa estimado 2D) es una estimación tronco-pelvis proyectada; no equivale a un COM segmentario 3D de laboratorio.")
                for title, cols in [
                    ("Pelvis y tronco", ["Oblicuidad pélvica","Inclinación lateral del tronco","Relación hombros-pelvis"]),
                    ("Hombros", ["Oblicuidad de hombros"]),
                    ("Cadera / rodilla · eje frontal", ["Rodilla frontal izquierda","Rodilla frontal derecha"]),
                    ("Pie · orientación distal", ["Orientación pie izquierda","Orientación pie derecha"]),
                    ("Pie · inclinación del retropié", ["Retropié izquierda","Retropié derecha"]),
                    ("Control frontal de rodilla", ["Valgo dinámico proyectado I","Valgo dinámico proyectado D"]),
                ]:
                    with st.expander(title, expanded=(title == "Pelvis y tronco")):
                        st.line_chart(chart_front.set_index("time_s")[cols], x_label="Tiempo (s)", y_label="Ángulo proyectado (°)")
            if chart_lat is not None:
                st.markdown("### Cámara 2 · lateral")
                for title, cols in [
                    ("Rodillas", ["Rodilla izquierda","Rodilla derecha"]),
                    ("Caderas", ["Cadera izquierda","Cadera derecha"]),
                    ("Tobillos / pie", ["Tobillo izquierda","Tobillo derecha"]),
                    ("Hombros", ["Hombro izquierda","Hombro derecha"]),
                ]:
                    with st.expander(title, expanded=(title == "Rodillas")):
                        st.line_chart(chart_lat.set_index("time_s")[cols], x_label="Tiempo (s)", y_label="Ángulo 2D (°)")
            if st.session_state.get("annotated_video_bytes") or st.session_state.get("annotated_video2_bytes"):
                st.markdown("### Vídeos con esqueleto + métricas")
                v1,v2 = st.columns(2)
                with v1:
                    if st.session_state.get("annotated_video_bytes"):
                        st.caption("Cámara 1 · frontal/posterior")
                        st.video(st.session_state.annotated_video_bytes)
                with v2:
                    if st.session_state.get("annotated_video2_bytes"):
                        st.caption("Cámara 2 · lateral")
                        st.video(st.session_state.annotated_video2_bytes)
        else:
            chart = st.session_state.chart
            c1,c2,c3,c4 = st.columns(4)
            c1.metric("Cadencia estimada", fmt(metric_value(metrics,"cadence_exp"),1)+" pasos/min")
            c2.metric("Regularidad temporal", fmt(metric_value(metrics,"regularity_cv"),1)+" % CV")
            c3.metric("Tracking válido", fmt(metric_value(metrics,"good_frames_pct"),1)+" %")
            c4.metric("Asimetría temporal", fmt(metric_value(metrics,"temporal_asymmetry_exp"),1)+" %")
            st.caption(f"Cadencia ({reference_text_for_metric('cadence_exp')}) · Variabilidad ({reference_text_for_metric('regularity_cv')}) · Asimetría ({reference_text_for_metric('temporal_asymmetry_exp')})")
            sl=metric_value(metrics,"stance_time_l_2d"); sr=metric_value(metrics,"stance_time_r_2d"); sa=metric_value(metrics,"stance_asymmetry_2d")
            qsl=metric_value(metrics,"support_segmentation_score_l"); qsr=metric_value(metrics,"support_segmentation_score_r")
            if sl is not None and sr is not None and np.isfinite(sl) and np.isfinite(sr):
                side="IZQUIERDA" if sl>sr else ("DERECHA" if sr>sl else "SIMILAR")
                st.info(f"Apoyo 2D por ciclos continuos: I {sl:.2f} s · D {sr:.2f} s · asimetría {fmt(sa,1)} % · mayor tiempo estimado: **{side}**. ({reference_text_for_metric('stance_asymmetry_2d')})")
            else:
                st.warning(f"Tiempo de apoyo NO informado: la segmentación apoyo/oscilación no superó el control temporal en ambos lados (I {fmt(qsl,0)}/100 · D {fmt(qsr,0)}/100). Se evita mostrar una duración potencialmente falsa.")
            st.markdown("#### Consistencia número de pasos · cadencia · duración")
            t1,t2,t3,t4 = st.columns(4)
            t1.metric("Eventos detectados", fmt(metric_value(metrics,"step_events_detected"),0))
            t2.metric("Duración segmento", fmt(metric_value(metrics,"segment_duration_s"),2)+" s")
            t3.metric("Eventos esperados", fmt(metric_value(metrics,"expected_steps_from_cadence"),1))
            t4.metric("Discrepancia", fmt(metric_value(metrics,"step_count_consistency_error_pct"),1)+" %")
            cq = metric_quality(metrics,"step_count_consistency_error_pct") or "No calculable"
            if cq == "Alta": st.success(f"Consistencia interna: {cq}")
            elif cq == "Moderada": st.info(f"Consistencia interna: {cq}")
            else: st.warning(f"Consistencia interna: {cq}. Conviene revisar visualmente los eventos detectados.")
            st.caption("Los eventos detectados son alternancias distales experimentales. La comprobación sirve para verificar coherencia interna, no para convertirlos automáticamente en heel-strikes clínicamente validados.")
            st.markdown("### Resumen biomecánico")
            for paragraph in biomech_summary(metrics, st.session_state.get("view", "")):
                st.write(paragraph)

            st.markdown("### Regularidad y consistencia paso a paso")
            rc1, rc2, rc3 = st.columns(3)
            rc1.metric("Consistencia paso a paso", fmt(metric_value(metrics,"step_to_step_similarity_index"),1)+" /100")
            rc2.metric("Cambio entre pasos", fmt(metric_value(metrics,"step_to_step_change_pct"),1)+" %")
            rc3.metric("CV temporal global", fmt(metric_value(metrics,"regularity_cv"),1)+" %")
            st.caption("Consistencia = cuánto se parece cada intervalo de paso al inmediatamente anterior. Simetría = cuánto se parecen izquierda y derecha. Son conceptos distintos.")

            if "Frontal" in st.session_state.get("view", ""):
                st.markdown("### Centro de masa estimado (COM (Centro de Masa estimado 2D)) y estabilidad dinámica")
                cc1,cc2,cc3,cc4 = st.columns(4)
                cc1.metric("Excursión ML", fmt(metric_value(metrics,"com_lateral_excursion_norm_pct"),1)+" % pelvis")
                cc2.metric("Excursión imagen", fmt(metric_value(metrics,"com_lateral_excursion_px"),1)+" px")
                cc3.metric("Velocidad ML P95", fmt(metric_value(metrics,"com_velocity_ml_px_s"),1)+" px/s")
                cc4.metric("COM (Centro de Masa estimado 2D) / BOS (Base de Sustentación) P95", fmt(metric_value(metrics,"com_bos_ratio_p95"),2))
                st.caption("COM (Centro de Masa estimado 2D) = centro corporal estimado a partir de pelvis y tronco. No equivale a un centro de masa segmentario 3D de laboratorio. La normalización por anchura pélvica mejora la comparación intraindividual.")
                _com_cols=[c for c in ["COM normalizado (% pelvis)","COM respecto BOS (ratio)"] if c in chart.columns]
                if _com_cols:
                    st.line_chart(chart.set_index("time_s")[_com_cols], x_label="Tiempo (s)", y_label="COM (Centro de Masa estimado 2D) normalizado")
                if "Velocidad COM ML (px/s)" in chart.columns:
                    with st.expander("Velocidad mediolateral del COM", expanded=False):
                        st.line_chart(chart.set_index("time_s")[["Velocidad COM ML (px/s)"]], x_label="Tiempo (s)", y_label="px/s")

            if "Frontal" in st.session_state.get("view", ""):
                st.markdown("### Biomecánica frontal/posterior 2D proyectada")
                st.warning("La rotación axial de cadera y la pronación son movimientos 3D. Aquí se muestran proxies 2D y no deben interpretarse como diagnóstico aislado.")
                for title, cols in [
                    ("Pelvis y tronco", ["Oblicuidad pélvica","Inclinación lateral del tronco","Relación hombros-pelvis"]),
                    ("Hombros", ["Oblicuidad de hombros"]),
                    ("Cadera / rodilla · eje frontal", ["Rodilla frontal izquierda","Rodilla frontal derecha"]),
                    ("Pie · orientación distal", ["Orientación pie izquierda","Orientación pie derecha"]),
                    ("Pie · inclinación del retropié", ["Retropié izquierda","Retropié derecha"]),
                    ("Control frontal de rodilla", ["Valgo dinámico proyectado I","Valgo dinámico proyectado D"]),
                ]:
                    with st.expander(title, expanded=(title == "Pelvis y tronco")):
                        st.line_chart(chart.set_index("time_s")[cols], x_label="Tiempo (s)", y_label="Ángulo proyectado (°)")
            else:
                st.markdown("### Cinemática sagital 2D proyectada")
                for title, cols in [
                    ("Rodillas", ["Rodilla izquierda","Rodilla derecha"]),
                    ("Caderas", ["Cadera izquierda","Cadera derecha"]),
                    ("Tobillos / pie", ["Tobillo izquierda","Tobillo derecha"]),
                    ("Hombros", ["Hombro izquierda","Hombro derecha"]),
                ]:
                    with st.expander(title, expanded=(title == "Rodillas")):
                        st.line_chart(chart.set_index("time_s")[cols], x_label="Tiempo (s)", y_label="Ángulo 2D (°)")
            if st.session_state.get("annotated_video_bytes"):
                st.markdown("### Vídeo con esqueleto + ángulos")
                st.video(st.session_state.annotated_video_bytes)
        with st.expander("Todas las métricas y calidad"):
            _mdf=pd.DataFrame(metrics).copy()
            _mdf["reference"]=_mdf["key"].map(reference_text_for_metric)
            st.dataframe(_mdf[["label","value","unit","quality","reference","notes"]], use_container_width=True, hide_index=True)
        st.caption("Cadencia, alternancia y sincronización automática permanecen experimentales hasta validación específica del protocolo.")

with tabs[4]:
    st.subheader("Pacientes / Evolución longitudinal")
    st.caption("Paciente → registros → comparación → evolución. Los vídeos no se almacenan; el histórico usa sesiones y métricas guardadas en Supabase.")
    if not sb_ready():
        st.info("Configura Supabase para activar el histórico persistente.")
    else:
        try:
            pats = sb_list_patients()
            if pats.empty:
                st.info("Todavía no hay pacientes en el histórico.")
            else:
                codes = pats.code.tolist()
                default = codes.index(patient) if patient in codes else 0
                selected = st.selectbox("Paciente / código", codes, index=default, key="history_patient")
                hist_all = sb_patient_history(selected)
                if hist_all.empty:
                    st.info("Este paciente todavía no tiene métricas guardadas.")
                else:
                    hist_all["created_dt"] = pd.to_datetime(hist_all["created_at"], utc=True)
                    hist_all["fecha"] = hist_all["created_dt"].dt.strftime("%d/%m/%Y %H:%M")
                    sessions = hist_all[[c for c in ["session_id","created_dt","fecha","record_name","mode","view","assistive_device","frontal_orientation","duration_s","duration_cam2_s","segment_start_s","segment_end_s","analysis_status","ready_3d"] if c in hist_all.columns]].drop_duplicates("session_id").sort_values("created_dt")

                    a,b,c,d = st.columns(4)
                    a.metric("Registros", sessions.session_id.nunique())
                    a0=sessions.created_dt.min(); a1=sessions.created_dt.max()
                    b.metric("Primer registro", a0.strftime("%d/%m/%Y") if pd.notna(a0) else "—")
                    c.metric("Último registro", a1.strftime("%d/%m/%Y") if pd.notna(a1) else "—")
                    d.metric("Seguimiento", f"{max(0,(a1-a0).days)} días" if pd.notna(a0) and pd.notna(a1) else "—")

                    st.markdown("### Línea de tiempo de registros")
                    timeline=sessions.copy()
                    timeline["Registro"] = timeline["record_name"].fillna("Marcha")
                    timeline["Fecha"] = timeline["fecha"]
                    timeline["Vista"] = timeline["view"].fillna("")
                    timeline["Ayuda"] = timeline["assistive_device"].fillna("Sin ayuda")
                    timeline["Estado"] = timeline["analysis_status"].fillna("")
                    st.dataframe(timeline[["Fecha","Registro","Vista","Ayuda","Estado"]], use_container_width=True, hide_index=True)

                    with st.expander("🗑️ Gestionar histórico: borrar registros o paciente", expanded=False):
                        st.caption(
                            "Puedes borrar uno o varios registros del paciente o eliminar completamente "
                            "el paciente y todo su histórico de Supabase. Los vídeos no están almacenados."
                        )
                        delete_mode = st.radio(
                            "Acción",
                            ["Borrar uno o varios registros", "Borrar paciente completo"],
                            horizontal=True,
                            key=f"history_delete_mode_{selected}",
                        )

                        if delete_mode == "Borrar uno o varios registros":
                            delete_labels = {
                                r["session_id"]: (
                                    f"{r['fecha']} · {r.get('record_name') or 'Marcha'} · "
                                    f"{r.get('view') or 'Vista no especificada'} · "
                                    f"{r.get('assistive_device') or 'Sin ayuda'}"
                                )
                                for _, r in sessions.sort_values("created_dt", ascending=False).iterrows()
                            }
                            delete_nonce = int(st.session_state.get(f"delete_nonce_{selected}", 0))
                            delete_ids = st.multiselect(
                                "Registros a borrar",
                                options=list(delete_labels.keys()),
                                format_func=lambda x: delete_labels[x],
                                key=f"delete_sessions_{selected}_{delete_nonce}",
                                placeholder="Selecciona uno o varios registros",
                            )
                            if delete_ids:
                                st.warning(
                                    f"Vas a borrar permanentemente {len(delete_ids)} registro(s) "
                                    "y todas sus métricas asociadas. Esta acción no se puede deshacer."
                                )
                                confirm_delete = st.checkbox(
                                    "Confirmo que quiero borrar permanentemente los registros seleccionados",
                                    key=f"confirm_delete_{selected}_{delete_nonce}",
                                )
                                if st.button(
                                    f"Eliminar {len(delete_ids)} registro(s)",
                                    type="primary",
                                    disabled=not confirm_delete,
                                    key=f"delete_button_{selected}_{delete_nonce}",
                                ):
                                    deleted = 0
                                    errors = []
                                    progress = st.progress(0.0, text="Eliminando y verificando en Supabase…")
                                    total_delete = len(delete_ids)
                                    for pos, sid in enumerate(delete_ids, start=1):
                                        try:
                                            result = sb_delete_session(sid)
                                            if result.get("verified"):
                                                deleted += 1
                                                # Si era la sesión activa, impedir cualquier reutilización posterior.
                                                if str(st.session_state.get("cloud_session_id") or "") == str(sid):
                                                    st.session_state.cloud_session_id = None
                                        except Exception as exc:
                                            errors.append(f"{delete_labels.get(sid, sid)}: {exc}")
                                        progress.progress(pos / total_delete, text=f"Verificados {pos}/{total_delete}")
                                    progress.empty()
                                    st.session_state[f"delete_nonce_{selected}"] = delete_nonce + 1

                                    # V67: verificación final de lote contra TODO el paciente lógico,
                                    # incluyendo posibles patient_id duplicados heredados. Solo se
                                    # confirma el borrado si ninguno de los session_id seleccionados
                                    # vuelve a aparecer en una lectura nueva de Supabase.
                                    try:
                                        fresh_hist = sb_patient_history(selected)
                                        remaining_ids = set()
                                        if isinstance(fresh_hist, pd.DataFrame) and not fresh_hist.empty and "session_id" in fresh_hist.columns:
                                            remaining_ids = set(fresh_hist["session_id"].dropna().astype(str).tolist())
                                        still_present = [str(sid) for sid in delete_ids if str(sid) in remaining_ids]
                                    except Exception as verify_exc:
                                        still_present = [str(sid) for sid in delete_ids]
                                        errors.append(f"Verificación final del lote: {verify_exc}")

                                    if still_present:
                                        st.error(
                                            "El borrado no se considera completado porque estos registros siguen "
                                            "presentes en Supabase tras la verificación final: " + ", ".join(still_present)
                                        )
                                    elif deleted == len(delete_ids) and not errors:
                                        st.success(f"Eliminación remota verificada: {deleted} registro(s).")
                                        st.rerun()
                                    else:
                                        if deleted:
                                            st.warning(f"Se eliminaron {deleted} de {len(delete_ids)} registro(s), pero el lote no quedó completamente verificado.")
                                        if errors:
                                            st.error("No se pudieron verificar todos los borrados:\n\n" + "\n".join(errors))

                        else:
                            n_sessions = int(sessions.session_id.nunique())
                            st.error(
                                f"Esta acción eliminará permanentemente el paciente **{selected}** y sus "
                                f"**{n_sessions} registro(s)**, incluidas todas las métricas asociadas."
                            )
                            patient_confirm_text = st.text_input(
                                f"Escribe {selected} para confirmar",
                                key=f"delete_patient_text_{selected}",
                            )
                            patient_confirm = patient_confirm_text.strip() == str(selected).strip()
                            if st.button(
                                "Eliminar paciente y todo su histórico",
                                type="primary",
                                disabled=not patient_confirm,
                                key=f"delete_patient_button_{selected}",
                            ):
                                try:
                                    with st.spinner("Eliminando paciente, registros y métricas; verificando en Supabase…"):
                                        result = sb_delete_patient(selected)
                                    if result.get("verified"):
                                        # Limpiar referencias locales a la sesión/paciente eliminado.
                                        st.session_state.cloud_session_id = None
                                        st.session_state.pop("history_patient", None)
                                        st.success(
                                            f"Paciente {selected} eliminado de forma verificada junto con "
                                            f"{result.get('deleted_sessions', 0)} registro(s)."
                                        )
                                        st.rerun()
                                except Exception as exc:
                                    st.error(f"No se pudo verificar el borrado completo del paciente: {exc}")

                    st.markdown("### Evolución de todos los registros")
                    f1,f2,f3 = st.columns(3)
                    aid_options=["Todas"]+sorted([x for x in hist_all.assistive_device.dropna().unique().tolist()]) if "assistive_device" in hist_all.columns else ["Todas"]
                    aid_filter=f1.selectbox("Ayuda técnica", aid_options, key="hist_aid")
                    view_options=["Todas"]+sorted([x for x in hist_all.view.dropna().unique().tolist()]) if "view" in hist_all.columns else ["Todas"]
                    view_filter=f2.selectbox("Vista", view_options, key="hist_view")
                    ref_mode=f3.selectbox("Referencia", ["Poblacional publicada", "Primer registro del paciente", "Sin referencia"], key="hist_ref")
                    hist=hist_all.copy()
                    if aid_filter!="Todas": hist=hist[hist.assistive_device==aid_filter]
                    if view_filter!="Todas": hist=hist[hist.view==view_filter]
                    labels = hist[["metric_key","metric_label","unit"]].drop_duplicates().sort_values("metric_label")
                    if labels.empty:
                        st.info("No hay métricas con esos filtros.")
                    else:
                        label = st.selectbox("Parámetro", labels.metric_label.tolist(), key="hist_metric")
                        key = labels.loc[labels.metric_label==label,"metric_key"].iloc[0]
                        h = hist[hist.metric_key==key].copy().sort_values("created_dt")
                        unit = h.unit.iloc[0] if len(h) else ""
                        if not h.empty:
                            plot=h[["fecha","value"]].dropna().set_index("fecha").rename(columns={"value":label})
                            st.line_chart(plot, x_label="Registro", y_label=f"{label} ({unit})" if unit else label)
                            first=float(h.value.dropna().iloc[0]) if h.value.notna().any() else np.nan
                            last=float(h.value.dropna().iloc[-1]) if h.value.notna().any() else np.nan
                            prev=float(h.value.dropna().iloc[-2]) if h.value.notna().sum()>=2 else np.nan
                            q1,q2,q3,q4=st.columns(4)
                            q1.metric("Basal", f"{first:.2f} {unit}" if np.isfinite(first) else "—")
                            q2.metric("Último", f"{last:.2f} {unit}" if np.isfinite(last) else "—", delta=f"{last-first:+.2f}" if np.isfinite(first) and np.isfinite(last) else None)
                            q3.metric("Δ vs basal", f"{((last-first)/abs(first)*100):+.1f} %" if np.isfinite(first) and first!=0 and np.isfinite(last) else "—")
                            q4.metric("Δ vs anterior", f"{last-prev:+.2f} {unit}" if np.isfinite(prev) and np.isfinite(last) else "—")
                            if ref_mode=="Poblacional publicada":
                                ref=reference_for_metric(key)
                                if ref:
                                    st.info(f"**Referencia contextual:** {ref['low']:.2f}–{ref['high']:.2f} {ref['unit']} · {reference_position(last,ref)}. {ref['population']}. {ref['applicability']}")
                                    st.caption(f"Fuente: {ref['source']} · DOI {ref['doi']} · Biblioteca PhysioSentinel {REFERENCE_LIBRARY_VERSION}")
                                else:
                                    st.warning("Esta métrica no tiene todavía una referencia poblacional suficientemente compatible con el método de PhysioSentinel. Se prioriza la evolución intraindividual.")
                            elif ref_mode=="Primer registro del paciente":
                                st.info("El primer registro filtrado se utiliza como referencia individual. Esto describe cambio respecto al basal y no implica por sí mismo mejoría o empeoramiento.")

                    st.markdown("### Comparar dos registros")
                    sess_labels={r.session_id:f"{r.fecha} · {r.record_name or 'Marcha'} · {r.view or ''} · {r.assistive_device or 'Sin ayuda'}" for _,r in sessions.iterrows()}
                    ids=list(sess_labels.keys())
                    if len(ids)<2:
                        st.info("Se necesitan al menos dos registros para una comparación directa.")
                    else:
                        ca,cb=st.columns(2)
                        sid_a=ca.selectbox("Registro A",ids,index=0,format_func=lambda x:sess_labels[x],key="cmp_a")
                        sid_b=cb.selectbox("Registro B",ids,index=len(ids)-1,format_func=lambda x:sess_labels[x],key="cmp_b")
                        if sid_a==sid_b:
                            st.warning("Selecciona dos registros diferentes.")
                        else:
                            ma=hist_all[hist_all.session_id==sid_a][["metric_key","metric_label","value","unit","quality"]].copy()
                            mb=hist_all[hist_all.session_id==sid_b][["metric_key","metric_label","value","unit","quality"]].copy()
                            cmp=ma.merge(mb,on="metric_key",suffixes=("_A","_B"))
                            cmp["Métrica"]=cmp.metric_label_A
                            cmp["A"]=cmp.value_A
                            cmp["B"]=cmp.value_B
                            cmp["Δ"]=cmp["B"]-cmp["A"]
                            cmp["Δ %"]=np.where(cmp["A"].abs()>1e-12,cmp["Δ"]/cmp["A"].abs()*100,np.nan)
                            cmp["Unidad"]=cmp.unit_A
                            st.dataframe(cmp[["Métrica","A","B","Δ","Δ %","Unidad"]].sort_values("Métrica"),use_container_width=True,hide_index=True)
                            sa=sessions[sessions.session_id==sid_a].iloc[0]; sb=sessions[sessions.session_id==sid_b].iloc[0]
                            comparable=(str(sa.get("view",""))==str(sb.get("view","")) and str(sa.get("assistive_device",""))==str(sb.get("assistive_device","")))
                            if comparable: st.success("Condiciones básicas comparables: misma vista y misma ayuda técnica.")
                            else: st.warning("Comparación contextual: cambia la vista y/o la ayuda técnica entre registros. Interpreta los Δ con cautela.")

                    st.markdown("### Resumen longitudinal")
                    st.write(f"**{selected}** dispone de **{sessions.session_id.nunique()} registros** entre {a0.strftime('%d/%m/%Y')} y {a1.strftime('%d/%m/%Y')}. La interpretación prioriza cambios intraindividuales y separa el valor medido de su significado clínico.")
                    st.caption("Un aumento o descenso no se etiqueta automáticamente como mejoría/empeoramiento. La dirección favorable depende de la variable, objetivo terapéutico, velocidad, ayuda técnica y contexto clínico.")
                    with st.expander("Biblioteca de referencias científicas"):
                        st.dataframe(pd.DataFrame(REFERENCE_SOURCES),use_container_width=True,hide_index=True)
                        st.caption("Las bandas solo se muestran cuando existe compatibilidad razonable con la métrica. No se trasladan rangos 3D a proxies 2D.")
        except Exception as e:
            st.error(f"No se pudo leer el histórico: {e}")

with tabs[5]:
    st.subheader("Redacción informe")
    st.caption("Informe clínico multimodal: espaciotemporales, 2D frontal/posterior, 2D lateral, integración biplanar y capas 3D disponibles. V61 añade una visión integrada final de todos los apartados.")
    if not st.session_state.get("metrics_done"):
        st.info("Calcula primero el análisis en la pestaña 3 y los resultados en la pestaña 4.")
    else:
        metrics=st.session_state.get("metrics",[])
        tech,patient_txt=generate_reports(
            metrics,
            st.session_state.get("view",view),
            st.session_state.get("patient_code", patient),
            st.session_state.get("record_name", record),
            st.session_state.get("assistive_device", assistive_device),
            patient_age=st.session_state.get("patient_age", 0),
            patient_sex=st.session_state.get("patient_sex", "No especificado"),
            record_date=st.session_state.get("record_date"),
        )
        st.markdown("### Informe de análisis biomecánico de la marcha (2D)")
        tech_edit=st.text_area("Texto técnico editable", value=tech, height=360, key="report_technical")
        st.markdown("### Versión simplificada para el paciente (complementaria)")
        pat_edit=st.text_area("Texto para paciente editable", value=patient_txt, height=320, key="report_patient")
        st.info("Convenio frontal documentado para pelvis: positivo (+) = lado izquierdo elevado / lado derecho descendido; negativo (-) = lado izquierdo descendido / lado derecho elevado. Para otras variables direccionales se evita atribuir signo anatómico cuando la orientación de cámara no permite hacerlo con seguridad.")
        st.markdown("### Visión integrada de los resultados · V61")
        _v61_integrated_default=_v61_integrated_text(metrics)
        _v61_integrated_edit=st.text_area("Integración multimodal editable", value=_v61_integrated_default, height=520, key="report_integrated_v61")
        st.caption("Integra los apartados previos sin convertir automáticamente los hallazgos en diagnóstico. Distingue 2D, biplanar, 3D estimado/no métrico y 3D calibrado.")
        st.markdown("### Rehabilitación · Objetivos derivados del análisis de la marcha · V62")
        _v62_rehab_default=_v62_rehabilitation_text(metrics)
        _v62_rehab_edit=st.text_area("Objetivos de rehabilitación editables", value=_v62_rehab_default, height=620, key="report_rehabilitation_v62")
        st.caption("Cada objetivo conserva trazabilidad con los hallazgos medidos y propone variables para reevaluación. La aplicación no atribuye automáticamente una desviación cinemática a debilidad, rigidez, dolor u otra etiología concreta.")
        full=f"PHYSIOSENTINEL GAIT {APP_VERSION}\n\nRESUMEN TÉCNICO-CLÍNICO\n{tech_edit}\n\nVISIÓN INTEGRADA DE LOS RESULTADOS\n{_v61_integrated_edit}\n\nREHABILITACIÓN · OBJETIVOS DERIVADOS DEL ANÁLISIS DE LA MARCHA\n{_v62_rehab_edit}\n\nINFORME PARA EL PACIENTE\n{pat_edit}\n"
        st.download_button("Descargar informe TXT", data=full.encode("utf-8"), file_name=f"PhysioSentinel_Gait_{patient}_{record}.txt", mime="text/plain")

with tabs[6]:
    st.subheader("Biplanar / Calibración · preparación 3D")
    st.write("La Versión 54 prepara el flujo para **triangulación 3D (Three-Dimensional / Tridimensional) con Pose2Sim** sin presentar resultados 3D como si estuvieran ya calculados.")
    st.markdown("### 1. Perfil de calibración")
    st.caption("Carga aquí un archivo TOML de calibración de las dos cámaras generado/validado para tu montaje. Se guarda como texto pequeño en Supabase; los vídeos siguen sin persistir.")
    cal_name = st.text_input("Nombre del perfil", value="Consulta_2cam")
    cal_notes = st.text_input("Notas", value="Frontal/posterior + lateral; cámaras fijas")
    cal_file = st.file_uploader("Archivo de calibración TOML", type=["toml"], key="calibration_toml")
    if cal_file is not None:
        try:
            content = cal_file.getvalue().decode("utf-8")
            toml.loads(content)
            st.success("El archivo tiene sintaxis TOML válida. Esto no sustituye la validación geométrica de la calibración.")
            if st.button("Guardar / actualizar perfil en Supabase", type="primary"):
                if not sb_ready():
                    st.error("Supabase no está configurado.")
                elif not cal_name.strip():
                    st.error("Escribe un nombre para el perfil.")
                else:
                    row = sb_upsert_calibration(cal_name.strip(), content, cal_notes, camera_count=2)
                    st.session_state.selected_calibration_name = cal_name.strip()
                    if st.session_state.get("cloud_session_id"):
                        sb_update_session_3d(st.session_state.cloud_session_id, calibration_profile_name=cal_name.strip())
                    st.success(f"Perfil **{cal_name.strip()}** guardado. Ya puede seleccionarse para sesiones de 2 cámaras.")
        except Exception as e:
            st.error(f"El archivo no es TOML válido: {e}")

    st.markdown("### 2. Perfiles disponibles")
    if sb_ready():
        try:
            rows = sb_list_calibrations()
            if rows:
                st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
            else:
                st.info("Todavía no hay perfiles de calibración guardados.")
        except Exception as e:
            st.warning(f"No se pudieron leer perfiles: {e}")

    st.markdown("### 3. Estado de preparación 3D de la sesión actual")
    q1 = st.session_state.get("camera1_quality")
    q2 = st.session_state.get("camera2_quality")
    sq = st.session_state.get("sync_quality", "No calculable")
    ready, reasons = readiness_3d(st.session_state.get("mode",""), q1, q2, sq, st.session_state.get("selected_calibration_name"))
    if ready:
        st.success("La sesión cumple los criterios de preparación de la Versión 54 para pasar a una futura triangulación 3D (Three-Dimensional / Tridimensional).")
    else:
        st.warning("Preparación 3D incompleta.")
        for reason in reasons:
            st.write(f"• {reason}")
    st.info("Siguiente etapa futura: materializar el perfil de calibración en /tmp, aplicar sincronización, ejecutar triangulación Pose2Sim, filtrar puntos 3D y posteriormente calcular cinemática/OpenSim. La Versión 54 todavía no ejecuta triangulación 3D (Three-Dimensional / Tridimensional) ni cinética estimada.")

with tabs[7]:
    st.subheader("Exportar / Descargar")
    st.caption(
        "Los gráficos, resultados y vídeos anotados se generan para descarga bajo demanda. "
        "No se guardan de forma permanente en Supabase, evitando acumular almacenamiento."
    )

    if not st.session_state.get("metrics_done"):
        st.info("Calcula primero los resultados de la sesión.")
    else:
        metrics = st.session_state.get("metrics", [])
        chart_single = st.session_state.get("chart")
        chart_front = st.session_state.get("chart_front")
        chart_lateral = st.session_state.get("chart_lateral")

        tech, patient_txt = generate_reports(
            metrics,
            st.session_state.get("view", view),
            st.session_state.get("patient_code", patient),
            st.session_state.get("record_name", record),
            st.session_state.get("assistive_device", assistive_device),
            patient_age=st.session_state.get("patient_age", 0),
            patient_sex=st.session_state.get("patient_sex", "No especificado"),
            record_date=st.session_state.get("record_date"),
        )

        # Si el usuario ha editado manualmente los textos del informe, exportar esos.
        tech_export = st.session_state.get("report_technical", tech)
        patient_export = st.session_state.get("report_patient", patient_txt)
        integrated_export = st.session_state.get("report_integrated_v61", _v61_integrated_text(metrics))
        rehabilitation_export = st.session_state.get("report_rehabilitation_v62", _v62_rehabilitation_text(metrics))

        st.markdown("### Contenido disponible")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Métricas", len(metrics))
        graph_cols = 0
        for cdf in (chart_single, chart_front, chart_lateral):
            if isinstance(cdf, pd.DataFrame):
                graph_cols += len([c for c in cdf.columns if c not in ("frame", "time_s")])
        c2.metric("Gráficos", graph_cols)
        c3.metric(
            "Vídeo anotado",
            "Sí" if st.session_state.get("annotated_video_bytes") else "No"
        )
        c4.metric(
            "2ª cámara",
            "Sí" if st.session_state.get("annotated_video2_bytes") else "No"
        )

        st.markdown("### Descargas individuales")

        metrics_df_export = _metrics_dataframe(metrics)
        if not metrics_df_export.empty:
            st.download_button(
                "⬇️ Resultados CSV",
                data=metrics_df_export.to_csv(index=False).encode("utf-8-sig"),
                file_name=f"resultados_{_safe_filename(st.session_state.get('patient_code','paciente'))}_{_safe_filename(st.session_state.get('record_name','marcha'))}.csv",
                mime="text/csv",
                use_container_width=True,
            )

        _integrated_export = st.session_state.get("report_integrated_v61", _v61_integrated_text(metrics))
        _rehabilitation_export = st.session_state.get("report_rehabilitation_v62", _v62_rehabilitation_text(metrics))
        report_full = (
            f"PHYSIOSENTINEL GAIT {APP_VERSION}\n\n"
            f"INFORME TÉCNICO\n{tech_export}\n\n"
            f"VISIÓN INTEGRADA DE LOS RESULTADOS\n{_integrated_export}\n\n"
            f"REHABILITACIÓN · OBJETIVOS DERIVADOS DEL ANÁLISIS DE LA MARCHA\n{_rehabilitation_export}\n\n"
            f"INFORME PARA EL PACIENTE\n{patient_export}\n"
        )
        st.download_button(
            "⬇️ Informes TXT",
            data=report_full.encode("utf-8"),
            file_name=f"informe_{_safe_filename(st.session_state.get('patient_code','paciente'))}_{_safe_filename(st.session_state.get('record_name','marcha'))}.txt",
            mime="text/plain",
            use_container_width=True,
        )

        if st.session_state.get("annotated_video_bytes"):
            st.download_button(
                "⬇️ Vídeo anotado cámara 1",
                data=st.session_state.annotated_video_bytes,
                file_name=f"video_anotado_cam01_{_safe_filename(st.session_state.get('patient_code','paciente'))}.mp4",
                mime="video/mp4",
                use_container_width=True,
            )
        if st.session_state.get("annotated_video2_bytes"):
            st.download_button(
                "⬇️ Vídeo anotado cámara 2",
                data=st.session_state.annotated_video2_bytes,
                file_name=f"video_anotado_cam02_{_safe_filename(st.session_state.get('patient_code','paciente'))}.mp4",
                mime="video/mp4",
                use_container_width=True,
            )

        st.markdown("### Paquete completo")
        st.write(
            "Incluye resultados CSV/JSON, informes, datos fuente de cada gráfico, "
            "todos los gráficos en PNG y los vídeos anotados disponibles."
        )

        try:
            export_zip = build_export_zip(
                metrics=metrics,
                chart=chart_single,
                chart_front=chart_front,
                chart_lateral=chart_lateral,
                technical_report=tech_export,
                patient_report=patient_export,
                annotated_video_bytes=st.session_state.get("annotated_video_bytes"),
                annotated_video2_bytes=st.session_state.get("annotated_video2_bytes"),
                patient_code=st.session_state.get("patient_code", patient),
                record_name=st.session_state.get("record_name", record),
                patient_age=st.session_state.get("patient_age", 0),
                patient_sex=st.session_state.get("patient_sex", "No especificado"),
                record_date=st.session_state.get("record_date"),
                app_version=APP_VERSION,
            )

            st.download_button(
                "📦 Descargar TODO en ZIP",
                data=export_zip,
                file_name=(
                    f"PhysioSentinel_exportacion_"
                    f"{_safe_filename(st.session_state.get('patient_code','paciente'))}_"
                    f"{_safe_filename(st.session_state.get('record_name','marcha'))}.zip"
                ),
                mime="application/zip",
                type="primary",
                use_container_width=True,
            )
            st.success(
                "Paquete preparado en memoria. Al descargarlo no se añade contenido pesado a la base de datos."
            )
        except Exception as e:
            st.error(f"No se pudo generar el paquete de exportación: {e}")
            with st.expander("Detalles técnicos"):
                st.code(traceback.format_exc())



def _compact_phase_cards(left_phase, right_phase, current_time_s, left_pct=np.nan, right_pct=np.nan):
    """v0.10.10 · Cabecera compacta para no sobredimensionar el nombre de fase."""
    def fmt_phase(phase, pct):
        pct_txt = f" · {pct:.0f}%" if np.isfinite(pct) else ""
        return f"{phase}{pct_txt}"

    st.markdown(
        f"""
        <style>
        .ps-phase-grid {{
            display:grid;
            grid-template-columns:1fr 1fr 0.65fr;
            gap:12px;
            margin:0 0 10px 0;
        }}
        .ps-phase-card {{
            border:1px solid rgba(128,128,128,.22);
            border-radius:10px;
            padding:9px 12px;
            min-width:0;
        }}
        .ps-phase-label {{
            font-size:0.76rem;
            opacity:.72;
            margin-bottom:2px;
        }}
        .ps-phase-value {{
            font-size:1.12rem;
            line-height:1.15;
            font-weight:650;
            white-space:normal;
            overflow-wrap:anywhere;
        }}
        .ps-time-value {{
            font-size:1.18rem;
            line-height:1.15;
            font-weight:700;
        }}
        </style>
        <div class="ps-phase-grid">
          <div class="ps-phase-card">
            <div class="ps-phase-label">Izquierda</div>
            <div class="ps-phase-value">{fmt_phase(left_phase,left_pct)}</div>
          </div>
          <div class="ps-phase-card">
            <div class="ps-phase-label">Derecha</div>
            <div class="ps-phase-value">{fmt_phase(right_phase,right_pct)}</div>
          </div>
          <div class="ps-phase-card">
            <div class="ps-phase-label">Tiempo</div>
            <div class="ps-time-value">{current_time_s:.2f} s</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


with tabs[8]:
    st.subheader("Analizador interactivo del ciclo de marcha")
    st.caption(
        "Vídeo + tracking + curvas cinemáticas + fases por extremidad sincronizadas. "
        "v0.10.12 fusiona fases y cinemática en una única figura con eje X compartido."
    )

    if not st.session_state.get("metrics_done"):
        st.info("Calcula primero los resultados en la pestaña 4.")
    else:
        # V50 · integración del MISMO ciclo físico en los dos planos 2D + auditoría TO.
        _v48=st.session_state.get("biplanar_v48_profile")
        if isinstance(_v48,dict) and isinstance(_v48.get("curves"),pd.DataFrame) and not _v48.get("curves").empty:
            st.markdown("### V52 · Perfil biomecánico biplanar del mismo ciclo físico")
            st.info("Frontal/posterior y lateral comparten aquí el mismo eje de fase 0–100 %. Cada curva mantiene su naturaleza 2D (Bidimensional): este panel NO triangula 3D (Tridimensional) ni fusiona ángulos de planos diferentes en una falsa variable espacial.")
            _s=_v48.get("summary") or {}; _c=_v48.get("curves")
            _a,_b,_d,_e=st.columns(4)
            _a.metric("Par homólogo",f"#{int(_s.get('selected_pair_index',0))}")
            _b.metric("Confianza",str(_s.get("integration_confidence","—")))
            _d.metric("Δ IC",f"{float(_s.get('ic_alignment_delta_ms',np.nan)):+.1f} ms" if np.isfinite(float(_s.get('ic_alignment_delta_ms',np.nan))) else "—")
            _e.metric("Δ duración",f"{float(_s.get('cycle_duration_delta_ms',np.nan)):+.1f} ms" if np.isfinite(float(_s.get('cycle_duration_delta_ms',np.nan))) else "—")
            st.caption(f"Extremidad frontal: {_s.get('front_side','?')} · lateral: {_s.get('lateral_side','?')} · hipótesis L/R invertida: {'sí' if _s.get('lr_swapped') else 'no'}. IC (Initial Contact / Contacto Inicial) = 0 % y siguiente IC = 100 %.")
            _to_stat=str(_s.get("to_concordance_status","No calculable"))
            _to_delta=float(_s.get("to_delta_pct",np.nan))
            _to_conf=float(_s.get("biplanar_to_confirmed_pct",np.nan))
            if _to_stat=="Concordante" and np.isfinite(_to_conf):
                st.success(f"TO (Toe Off / Despegue del Pie) biplanar concordante: Δ {_to_delta:.1f} puntos porcentuales · evento confirmado ≈ {_to_conf:.1f} % del ciclo.")
            elif _to_stat=="Concordancia limitada":
                st.warning(f"TO frontal y lateral presentan concordancia limitada (Δ {_to_delta:.1f} puntos porcentuales). V50 mantiene ambos detectores por separado, pero NO genera un TO biplanar confirmado.")
            elif _to_stat=="Discrepante":
                st.error(f"TO frontal y lateral son discrepantes (Δ {_to_delta:.1f} puntos porcentuales). El IC/ciclo físico sigue siendo válido, pero V50 NO utiliza el TO como evento biplanar confirmado.")
            else:
                st.info("TO (Toe Off / Despegue del Pie) biplanar no calculable; el ciclo se mantiene anclado por IC→IC.")
            st.markdown("#### V50 · Auditoría multiseñal del TO (Toe Off / Despegue del Pie)")
            _v50fp=float(_s.get("v50_front_to_candidate_pct",np.nan)); _v50lp=float(_s.get("v50_lateral_to_candidate_pct",np.nan))
            _v50d=float(_s.get("v50_candidate_delta_pct",np.nan)); _v50c=float(_s.get("v50_biplanar_to_confirmed_pct",np.nan))
            _v50stat=str(_s.get("v50_to_status","No calculable")); _v50q=str(_s.get("v50_to_quality","No calculable"))
            _t1,_t2,_t3,_t4=st.columns(4)
            _t1.metric("TO frontal V50",f"{_v50fp:.1f} %" if np.isfinite(_v50fp) else "—")
            _t2.metric("TO lateral V50",f"{_v50lp:.1f} %" if np.isfinite(_v50lp) else "—")
            _t3.metric("Δ candidatos",f"{_v50d:.1f} pp" if np.isfinite(_v50d) else "—")
            _t4.metric("TO biplanar V50",f"{_v50c:.1f} %" if np.isfinite(_v50c) else "No confirmado")
            if _v50stat=="Confirmado biplanar" and np.isfinite(_v50c):
                st.success(f"V50 confirma TO biplanar ≈ {_v50c:.1f}% del ciclo · calidad {_v50q}. El reloj V47 y los IC no se modifican.")
            elif _v50stat=="Respaldado lateralmente; no confirmado biplanar":
                st.warning("V50 encuentra respaldo interno para el TO lateral, pero la vista frontal no converge lo suficiente. Se conserva como referencia lateral, NO como TO biplanar confirmado.")
            elif "Discrepante" in _v50stat:
                st.error("V50 mantiene discrepancia entre vistas. No fuerza promedio ni desplaza el reloj físico; el TO permanece no confirmado biplanar.")
            else:
                st.info(f"Estado V50 del TO: {_v50stat} · calidad {_v50q}.")
            st.markdown("#### V51 · Detector TO jerárquico dependiente de perspectiva")
            _v51sel=float(_s.get("v51_selected_to_pct",np.nan)); _v51sup=float(_s.get("v51_front_support_at_lateral_to",np.nan))
            _v51conf=float(_s.get("v51_to_confirmed",0.0)); _v51stat=str(_s.get("v51_to_status","No calculable")); _v51q=str(_s.get("v51_to_quality","No calculable"))
            _v51unc=float(_s.get("v51_selected_to_uncertainty_ms",np.nan)); _v51role=str(_s.get("v51_front_role","—"))
            _h1,_h2,_h3,_h4=st.columns(4)
            _h1.metric("Vista primaria","Lateral/sagital")
            _h2.metric("TO seleccionado V51",f"{_v51sel:.1f} %" if np.isfinite(_v51sel) else "—")
            _h3.metric("Respaldo frontal",f"{_v51sup:.2f}" if np.isfinite(_v51sup) else "—")
            _h4.metric("Resolución temporal",f"±{_v51unc:.1f} ms" if np.isfinite(_v51unc) else "—")
            if _v51conf>=0.5 and _v51stat=="Confirmado biplanar directo":
                st.success(f"V51 confirma TO de forma biplanar directa en ≈ {_v51sel:.1f}% del ciclo · calidad {_v51q}. La lateral gobierna el instante y la frontal presenta candidato concordante.")
            elif _v51conf>=0.5 and _v51stat=="Confirmado jerárquico":
                st.success(f"V51 confirma TO de forma jerárquica en ≈ {_v51sel:.1f}% del ciclo · calidad {_v51q}. La vista lateral es primaria y la frontal aporta evidencia compatible en ese mismo instante ({_v51role}).")
            elif "probable" in _v51stat.lower():
                st.warning(f"V51 conserva TO lateral ≈ {_v51sel:.1f}% como probable, pero la evidencia frontal es no concluyente. No se etiqueta como TO biplanar confirmado.")
            else:
                st.warning(f"Estado V51 del TO: {_v51stat} · calidad {_v51q}. No se fuerza promedio entre vistas.")
            st.caption("Criterio V51: la vista lateral/sagital tiene prioridad para TO; la frontal/posterior verifica compatibilidad. Los umbrales son internos y experimentales 2D, no equivalen a una plataforma de fuerzas ni a validación cinética.")
            with st.expander("Auditoría V51 · decisión jerárquica del TO",expanded=False):
                _v51a=_v48.get("v51_to_audit")
                if isinstance(_v51a,pd.DataFrame) and not _v51a.empty:
                    st.dataframe(_v51a,use_container_width=True,hide_index=True)
                else:
                    st.caption("Auditoría V51 no disponible para este ciclo.")

            st.markdown("#### V52 · Ciclo biomecánico biplanar clínico IC→TO→IC")
            _v52st=str(_s.get("v52_status","No calculable")); _v52q=str(_s.get("v52_quality","No calculable"))
            _v52ap=float(_s.get("v52_stance_pct",np.nan)); _v52sw=float(_s.get("v52_swing_pct",np.nan))
            _v52aps=float(_s.get("v52_stance_duration_s",np.nan)); _v52sws=float(_s.get("v52_swing_duration_s",np.nan))
            _v52c1,_v52c2,_v52c3,_v52c4=st.columns(4)
            _v52c1.metric("Apoyo IC→TO",f"{_v52ap:.1f} %" if np.isfinite(_v52ap) else "—", f"{_v52aps:.2f} s" if np.isfinite(_v52aps) else None)
            _v52c2.metric("TO",f"{_v51sel:.1f} %" if np.isfinite(_v51sel) else "—")
            _v52c3.metric("Oscilación TO→IC",f"{_v52sw:.1f} %" if np.isfinite(_v52sw) else "—", f"{_v52sws:.2f} s" if np.isfinite(_v52sws) else None)
            _v52c4.metric("Calidad ciclo clínico",_v52q)
            if _v52st=="Ciclo clínico completo":
                st.success(f"V52 estructura el mismo ciclo físico como IC → apoyo ({_v52ap:.1f} %) → TO ({_v51sel:.1f} %) → oscilación ({_v52sw:.1f} %) → siguiente IC. No recalcula V47 ni V51.")
            else:
                st.warning(f"Estado V52: {_v52st}. La división apoyo/oscilación se muestra con la calidad heredada del TO V51 ({_v52q}).")
            _fig52=_v52_timeline_figure(_v51sel)
            if _fig52 is not None:
                st.pyplot(_fig52,use_container_width=True); plt.close(_fig52)
            _v52pm=_v48.get("v52_phase_metrics")
            with st.expander("V52 · Resumen de las 8 curvas por apoyo y oscilación",expanded=False):
                if isinstance(_v52pm,pd.DataFrame) and not _v52pm.empty:
                    st.dataframe(_v52pm,use_container_width=True,hide_index=True)
                    st.caption("Media, mínimo, máximo y ROM (Rango de Movimiento) se calculan dentro de cada fase del mismo ciclo. Son descriptores 2D y no rangos normativos.")
                else:
                    st.caption("No hay métricas V52 por fase disponibles para este ciclo.")
            st.caption("V52 es una capa clínica descriptiva: organiza las curvas 2D en apoyo/oscilación sobre el mismo ciclo físico. No añade fuerzas, momentos articulares ni reconstrucción 3D.")

            with st.expander("Auditoría V50 · señales usadas para localizar el onset distal",expanded=False):
                _v50a=_v48.get("v50_to_audit")
                if isinstance(_v50a,pd.DataFrame) and not _v50a.empty:
                    st.dataframe(_v50a,use_container_width=True,hide_index=True)
                    st.caption("activation combina velocidad distal, ascenso del pie y, en lateral, avance del pie respecto a pelvis. Se busca el inicio persistente de activación, no el máximo tardío de oscilación.")
                else:
                    st.caption("Auditoría multiseñal no disponible para este ciclo.")
            _front_cols=[x for x in _c.columns if str(x).startswith("Frontal ·")]
            _lat_cols=[x for x in _c.columns if str(x).startswith("Lateral ·")]
            st.caption(f"Curvas integradas: frontal/posterior {len(_front_cols)} · lateral/sagital {len(_lat_cols)}.")
            if not _front_cols:
                st.warning("No se han podido materializar curvas frontales en este ciclo. V52 no presenta el perfil como biplanar completo hasta recuperar al menos una variable frontal.")
            _fc,_lc=st.columns(2,gap="large")
            with _fc:
                _figf=_v48_profile_figure(_c,_front_cols,"Plano frontal/posterior · ciclo homólogo",float(_s.get("front_to_pct",np.nan)),np.nan)
                if _figf is not None:
                    st.pyplot(_figf,use_container_width=True); plt.close(_figf)
            with _lc:
                _figl=_v48_profile_figure(_c,_lat_cols,"Plano sagital/lateral · ciclo homólogo",np.nan,float(_s.get("lateral_to_pct",np.nan)))
                if _figl is not None:
                    st.pyplot(_figl,use_container_width=True); plt.close(_figl)
            with st.expander("Datos V52 · ciclo normalizado 0–100 %",expanded=False):
                st.dataframe(_c,use_container_width=True,hide_index=True)
                st.caption("Las curvas exportables corresponden al mismo ciclo físico seleccionado por el emparejamiento V47; la normalización temporal facilita comparar eventos y cinemática entre planos sin afirmar reconstrucción 3D.")
            st.divider()

        # Selección de snapshot. En 2 cámaras puede alternarse entre frontal y lateral.
        two_cycle_views = (
            _valid_dataframe(st.session_state.get("cycle_seg_front"))
            and _valid_dataframe(st.session_state.get("cycle_seg_lateral"))
        )
        cycle_view_choice = "Frontal/posterior"
        if two_cycle_views:
            cycle_view_choice = st.radio(
                "Vista para revisar el ciclo",
                ["Frontal/posterior", "Lateral"],
                horizontal=True,
                key="cycle_review_view",
            )

        if cycle_view_choice == "Lateral" and two_cycle_views:
            seg_cycle = st.session_state.get("cycle_seg_lateral")
            _fps_lat = st.session_state.get("cycle_fps_lateral")
            fps_cycle = float(_fps_lat if _fps_lat is not None else 30.0)
            Lsum = st.session_state.get("cycle_support_left_lateral")
            Rsum = st.session_state.get("cycle_support_right_lateral")
            clean_bytes_cycle = st.session_state.get("cycle_video2_clean_bytes")
            annotated_bytes_cycle = st.session_state.get("annotated_video2_bytes")
            selected_subject_cycle = st.session_state.get("selected_subject_label_cam2")
        else:
            seg_cycle = st.session_state.get("cycle_seg")
            if seg_cycle is None:
                seg_cycle = st.session_state.get("cycle_seg_front")

            fps_cycle = st.session_state.get("cycle_fps")
            if fps_cycle is None:
                fps_cycle = st.session_state.get("cycle_fps_front")
            fps_cycle = float(fps_cycle if fps_cycle is not None else 30.0)

            Lsum = st.session_state.get("cycle_support_left")
            if Lsum is None:
                Lsum = st.session_state.get("cycle_support_left_front")

            Rsum = st.session_state.get("cycle_support_right")
            if Rsum is None:
                Rsum = st.session_state.get("cycle_support_right_front")

            clean_bytes_cycle = st.session_state.get("cycle_video_clean_bytes")
            annotated_bytes_cycle = st.session_state.get("annotated_video_bytes")
            selected_subject_cycle = st.session_state.get("selected_subject_label_cam1")

        # Fallback de compatibilidad: si todavía existe analysis_df, reconstruir una vez.
        if not _valid_dataframe(seg_cycle):
            df_fallback = st.session_state.get("analysis_df")
            if _valid_dataframe(df_fallback):
                seg_cycle = df_fallback.copy()
                snap_fb = _make_cycle_snapshot(seg_cycle, fps_cycle)
                if snap_fb:
                    seg_cycle = snap_fb["seg"]
                    Lsum = snap_fb["left"]
                    Rsum = snap_fb["right"]
                    st.session_state.cycle_seg = seg_cycle
                    st.session_state.cycle_support_left = Lsum
                    st.session_state.cycle_support_right = Rsum
                    st.session_state.cycle_fps = fps_cycle

        if not _valid_dataframe(seg_cycle):
            st.warning(
                "No hay snapshot del segmento 2D. Este registro fue calculado antes de v0.10.1 "
                "o se perdió la sesión de Streamlit. Recalcula una vez el registro; a partir de entonces "
                "la pestaña 9 conservará el segmento aunque se eliminen los vídeos/JSON temporales."
            )
        else:
            try:
                if not isinstance(Lsum, dict) or not isinstance(Rsum, dict):
                    snap_rebuild = _make_cycle_snapshot(seg_cycle, fps_cycle)
                    Lsum = snap_rebuild["left"] if snap_rebuild else {}
                    Rsum = snap_rebuild["right"] if snap_rebuild else {}

                left_cycles = _build_side_cycles(Lsum, fps_cycle, "L")
                right_cycles = _build_side_cycles(Rsum, fps_cycle, "R")

                if selected_subject_cycle:
                    st.success(
                        f"🔒 Sujeto biomecánico bloqueado: **{selected_subject_cycle}**"
                    )

                if not left_cycles and not right_cycles:
                    st.warning("No se han podido construir ciclos IC→TO→IC válidos.")
                else:
                    # v0.10.7 · navegación en SEGUNDOS y mapa temporal de fases.
                    frame_series = pd.to_numeric(seg_cycle["frame"], errors="coerce")
                    fmin = int(np.nanmin(frame_series))
                    fmax = int(np.nanmax(frame_series))
                    duration_s = max(0.0, (fmax-fmin)/fps_cycle)

                    # Estado maestro en segundos para que la barra tenga significado clínico.
                    time_key = "cycle_master_time_s"
                    if time_key not in st.session_state:
                        st.session_state[time_key] = 0.0
                    st.session_state[time_key] = float(
                        np.clip(st.session_state[time_key], 0.0, duration_s)
                    )

                    # Barra MAESTRA bajo el vídeo.
                    current_time_s = float(st.session_state[time_key])
                    current_frame = int(round(fmin + current_time_s*fps_cycle))
                    current_frame = int(np.clip(current_frame, fmin, fmax))

                    left_cycle = _find_cycle_for_frame(left_cycles, current_frame)
                    right_cycle = _find_cycle_for_frame(right_cycles, current_frame)
                    left_phase, left_pct = _phase_context_text(left_cycles, current_frame)
                    right_phase, right_pct = _phase_context_text(right_cycles, current_frame)

                    video_col, phase_col = st.columns([0.72,1.28], gap="medium")

                    with video_col:
                        st.caption("Vídeo + tracking")
                        frame_bytes = None
                        if clean_bytes_cycle:
                            clean_frame = _video_frame_from_bytes(clean_bytes_cycle, current_frame)
                            frame_bytes = _draw_selected_tracking(
                                clean_frame, current_frame, seg_cycle,
                                left_cycle, right_cycle, crop_subject=True
                            )
                        else:
                            video_path_cycle = (
                                st.session_state.get("video2_path")
                                if cycle_view_choice=="Lateral"
                                else st.session_state.get("video1_path")
                            )
                            if video_path_cycle and Path(video_path_cycle).exists():
                                cap_tmp=cv2.VideoCapture(str(video_path_cycle))
                                cap_tmp.set(cv2.CAP_PROP_POS_FRAMES,current_frame)
                                ok_tmp,clean_frame=cap_tmp.read()
                                cap_tmp.release()
                                if ok_tmp:
                                    frame_bytes=_draw_selected_tracking(
                                        clean_frame,current_frame,seg_cycle,
                                        left_cycle,right_cycle,crop_subject=True
                                    )

                        if frame_bytes:
                            st.image(frame_bytes, width=360)
                        else:
                            st.info("Vídeo no disponible; las curvas permanecen accesibles.")

                        st.caption(
                            f"Frame {current_frame} · {current_time_s:.2f} s"
                        )

                        # Navegación rápida por eventos anatómicos.
                        events = _all_event_frames(left_cycles, right_cycles)
                        prev_events = [e for e in events if e[0] < current_frame]
                        next_events = [e for e in events if e[0] > current_frame]
                        b1,b2 = st.columns(2)
                        if b1.button("◀ Evento anterior", use_container_width=True):
                            if prev_events:
                                target = prev_events[-1][0]
                                st.session_state[time_key] = (target-fmin)/fps_cycle
                                st.rerun()
                        if b2.button("Evento siguiente ▶", use_container_width=True):
                            if next_events:
                                target = next_events[0][0]
                                st.session_state[time_key] = (target-fmin)/fps_cycle
                                st.rerun()

                        # Contexto textual inmediato.
                        left_phase, left_pct = _phase_context_text(left_cycles, current_frame)
                        right_phase, right_pct = _phase_context_text(right_cycles, current_frame)
                        st.caption(
                            f"{current_time_s:.2f} s · frame {current_frame}  |  "
                            f"Izq: {left_phase}"
                            + (f" ({left_pct:.0f}%)" if np.isfinite(left_pct) else "")
                            + "  ·  "
                            f"Der: {right_phase}"
                            + (f" ({right_pct:.0f}%)" if np.isfinite(right_pct) else "")
                        )

                    # Recalcular tras el widget.
                    left_cycle = _find_cycle_for_frame(left_cycles, current_frame)
                    right_cycle = _find_cycle_for_frame(right_cycles, current_frame)
                    left_phase, left_pct = _phase_context_text(left_cycles, current_frame)
                    right_phase, right_pct = _phase_context_text(right_cycles, current_frame)

                    with phase_col:
                        _compact_phase_cards(
                            left_phase, right_phase, current_time_s,
                            left_pct=left_pct, right_pct=right_pct
                        )

                        # Ajusta la geometría del slider para que el track útil
                        # coincida con el área horizontal de los gráficos.
                        st.markdown(
                            """
                            <style>
                            .st-key-cycle_master_time_s {
                                margin-left: 5.6%;
                                width: 93.4%;
                            }
                            </style>
                            """,
                            unsafe_allow_html=True,
                        )

                        st.markdown("**Cursor temporal maestro**")
                        st.caption(
                            "El control y ambos paneles comparten el mismo tiempo. "
                            "v0.10.12 dibuja fases y cinemática dentro de una única figura "
                            "con un eje X compartido."
                        )
                        current_time_s = st.slider(
                            "Seleccionar instante",
                            min_value=0.0,
                            max_value=float(duration_s),
                            value=float(st.session_state[time_key]),
                            step=float(1.0/fps_cycle),
                            format="%.2f s",
                            key=time_key,
                            label_visibility="collapsed",
                            help="Control único del vídeo y del cursor de ambos paneles."
                        )

                        current_frame = int(round(fmin + current_time_s*fps_cycle))
                        current_frame = int(np.clip(current_frame, fmin, fmax))
                        left_cycle = _find_cycle_for_frame(left_cycles, current_frame)
                        right_cycle = _find_cycle_for_frame(right_cycles, current_frame)
                        left_phase, left_pct = _phase_context_text(left_cycles, current_frame)
                        right_phase, right_pct = _phase_context_text(right_cycles, current_frame)

                        kin = _compute_display_kinematics(seg_cycle)
                        numeric_candidates = [
                            c for c in kin.columns
                            if c not in ("frame","time_s")
                            and pd.api.types.is_numeric_dtype(kin[c])
                            and any(k in c for k in (
                                "Oblicuidad","Flexión","Ángulo tobillo","Inclinación",
                                "Desviación frontal","Orientación pie","Retropié"
                            ))
                            and pd.to_numeric(kin[c],errors="coerce").notna().sum() >= 2
                        ]

                        if not numeric_candidates:
                            st.warning("No hay señales cinemáticas suficientes para graficar.")
                        else:
                            dli = next((i for i,c in enumerate(numeric_candidates) if "Izquierda" in c),0)
                            dri = next((i for i,c in enumerate(numeric_candidates) if "Derecha" in c),min(1,len(numeric_candidates)-1))
                            q1,q2 = st.columns(2)
                            var_l = q1.selectbox("Curva izquierda", numeric_candidates, index=dli, key="cycle_var_l")
                            var_r = q2.selectbox("Curva derecha", numeric_candidates, index=dri, key="cycle_var_r")

                            # v0.10.12 · una sola figura = alineación real garantizada.
                            fig_sync = _combined_phase_kinematic_figure(
                                left_cycles, right_cycles, kin,
                                var_l, var_r,
                                fps_cycle, fmin, fmax, current_frame
                            )
                            st.pyplot(fig_sync, use_container_width=True)

                            bio_sync = io.BytesIO()
                            fig_sync.savefig(bio_sync,format="png",dpi=170,bbox_inches="tight")
                            plt.close(fig_sync)
                            st.session_state["cycle_phase_png"] = bio_sync.getvalue()
                            st.session_state["cycle_kinematic_png"] = bio_sync.getvalue()

                            with st.expander("Curva normalizada 0–100% del ciclo actual", expanded=False):
                                df_l = _normalized_cycle_series(kin,left_cycle,[var_l]) if left_cycle else pd.DataFrame()
                                df_r = _normalized_cycle_series(kin,right_cycle,[var_r]) if right_cycle else pd.DataFrame()
                                fig_kin_norm = _kinematic_cycle_figure(
                                    df_l,df_r,var_l,var_r,
                                    left_pct=left_pct,right_pct=right_pct
                                )
                                st.pyplot(fig_kin_norm,use_container_width=True)
                                plt.close(fig_kin_norm)


                    st.markdown("### Reproducción continua")
                    st.caption(
                        "Reproduce el segmento completo con las fases izquierda/derecha "
                        "sobreimpresas. Selecciona una velocidad para cámara lenta o rápida."
                    )

                    speed_col, action_col = st.columns([0.45,0.55])
                    with speed_col:
                        playback_speed = st.selectbox(
                            "Velocidad",
                            [0.25,0.5,1.0,1.5,2.0],
                            index=2,
                            format_func=lambda x: f"{x:g}×",
                            key="cycle_playback_speed",
                        )
                    with action_col:
                        st.write("")
                        prepare_playback = st.button(
                            "▶ Preparar reproducción",
                            use_container_width=True,
                            key="prepare_cycle_playback",
                        )

                    # Construir la versión base solo cuando se solicita.
                    if prepare_playback:
                        source_playback = annotated_bytes_cycle or clean_bytes_cycle
                        if source_playback:
                            with st.spinner("Preparando vídeo de revisión…"):
                                base_phase_video = _build_phase_review_video_bytes(
                                    source_playback, left_cycles, right_cycles, fps_cycle
                                )
                                st.session_state["cycle_phase_review_base"] = base_phase_video
                                st.session_state["cycle_phase_review_speed"] = None
                        else:
                            st.warning("No queda vídeo disponible en memoria para preparar la reproducción.")

                    base_phase_video = st.session_state.get("cycle_phase_review_base")
                    if base_phase_video:
                        cached_speed = st.session_state.get("cycle_phase_review_speed")
                        if cached_speed != playback_speed or not st.session_state.get("cycle_phase_review_bytes"):
                            with st.spinner(f"Preparando reproducción a {playback_speed:g}×…"):
                                speed_bytes = _change_video_speed_bytes(
                                    base_phase_video, playback_speed
                                )
                                st.session_state["cycle_phase_review_bytes"] = speed_bytes or base_phase_video
                                st.session_state["cycle_phase_review_speed"] = playback_speed

                        st.video(st.session_state["cycle_phase_review_bytes"])
                        st.caption(
                            "Este reproductor es para inspección continua. "
                            "Para comprobar un instante exacto usa la barra temporal superior."
                        )

                    with st.expander("⚙️ Corrección manual del ciclo actual",expanded=False):
                        st.caption("Solo si IC/TO no coincide visualmente. No vuelve a ejecutar Pose2Sim.")
                        e1,e2=st.columns(2)
                        with e1:
                            st.markdown("**Izquierda**")
                            left_cycle=_manual_event_editor(left_cycle,f"manual_L_auto_{left_cycle['cycle_index'] if left_cycle else 0}") if left_cycle else None
                        with e2:
                            st.markdown("**Derecha**")
                            right_cycle=_manual_event_editor(right_cycle,f"manual_R_auto_{right_cycle['cycle_index'] if right_cycle else 0}") if right_cycle else None

                        with st.expander("Resumen numérico del ciclo actual", expanded=False):

                            # Table with phase summary
                            rows=[]
                            for cyc,label in [(left_cycle,"Izquierda"),(right_cycle,"Derecha")]:
                                if not cyc:
                                    continue
                                cyc_frames=max(1,cyc["next_ic_frame"]-cyc["ic_frame"])
                                stance_pct=float(cyc.get("stance_pct",np.nan))
                                swing_pct=100.0-stance_pct if np.isfinite(stance_pct) else np.nan
                                rows.append({
                                    "Extremidad":label,
                                    "IC frame":cyc["ic_frame"],
                                    "TO frame":cyc.get("to_frame"),
                                    "IC siguiente":cyc["next_ic_frame"],
                                    "Ciclo (s)":cyc_frames/fps_cycle,
                                    "Apoyo (%)":stance_pct,
                                    "Swing (%)":swing_pct,
                                    "Apoyo (s)":((cyc.get("to_frame")-cyc["ic_frame"])/fps_cycle) if cyc.get("to_frame") else np.nan,
                                    "Swing (s)":((cyc["next_ic_frame"]-cyc.get("to_frame"))/fps_cycle) if cyc.get("to_frame") else np.nan,
                                })
                            if rows:
                                phase_df=pd.DataFrame(rows)
                                st.dataframe(phase_df,use_container_width=True,hide_index=True)
                                st.session_state["cycle_phase_table"]=phase_df

                        st.caption(
                            "IC/TO = eventos cinemáticos 2D estimados; no equivalen a fuerza de impacto."
                        )

            except Exception as e:
                st.error(f"No se pudo construir el analizador de ciclo: {e}")
                with st.expander("Detalles técnicos"):
                    st.code(traceback.format_exc())



with tabs[9]:
    st.subheader("Capacidad funcional de marcha + Análisis biomecánico por tracking")
    st.caption(
        "Desde la Versión 29 la capacidad locomotora y los perfiles biomecánicos se calculan por separado. "
        "El perfil biomecánico cambia según la vista disponible para evitar penalizar variables que una cámara concreta no puede interpretar con suficiente validez."
    )

    metrics_idx=st.session_state.get("metrics")
    if not metrics_idx:
        st.info("Calcula primero los resultados de la marcha en la pestaña 4.")
    else:
        idx=compute_biomechanical_profile(metrics_idx,st.session_state.get("mode",""),st.session_state.get("view",""))
        loc=compute_locomotor_capacity_index(
            metrics_idx,
            st.session_state.get("functional_support","Independiente"),
            st.session_state.get("measured_gait_speed",0.0),
        )

        st.markdown("### 1. Capacidad locomotora")
        l1,l2,l3=st.columns([1.0,1.5,1.0])
        lscore=loc["score"]
        l1.metric("Índice de capacidad locomotora",f"{lscore:.0f}/100" if np.isfinite(lscore) else "—")
        l2.metric("Clasificación funcional",loc["classification"])
        l3.metric("Confianza de medición",loc["confidence_label"])
        if np.isfinite(lscore): st.progress(int(np.clip(round(lscore),0,100)))
        if loc["caps"]:
            st.warning("**Regla de déficit dominante activada:** "+"; ".join(loc["caps"])+". Una buena alineación segmentaria no puede compensar este déficit funcional.")
        if loc["confidence_label"]=="Limitada":
            st.warning("La confianza del índice funcional es limitada por controles de calidad/detección. Esto reduce la certeza de la medida, pero NO se contabiliza como peor función del paciente.")
        st.caption("ICLM (Índice de Capacidad Locomotora) 0.1 experimental. La velocidad entra solo si existe una medida válida en m/s; el apoyo externo se registra clínicamente y no se infiere automáticamente del vídeo.")

        st.markdown("### 2. Análisis biomecánico por tracking y plano de movimiento")
        score=idx["global_score"]
        b1,b2,b3=st.columns([1.25,1.0,1.0])
        b1.metric(idx["profile_name"],f"{score:.0f}/100" if np.isfinite(score) else "—")
        b2.metric("Cobertura",f"{idx['coverage_pct']:.0f}%")
        b3.metric("Confianza de medición",idx["confidence_label"])
        st.markdown(f"**Clasificación biomecánica:** {idx['classification']}")
        st.info(idx["method_note"])
        if idx.get("frontal_caps"):
            st.warning("**Regla frontal no compensatoria activada:** "+"; ".join(idx["frontal_caps"])+". Un dominio postural conservado no puede ocultar varios déficits frontales marcados.")
        if idx.get("subprofiles"):
            _spcols=st.columns(len(idx["subprofiles"]))
            for _col,(_spname,_sp) in zip(_spcols,idx["subprofiles"].items()):
                _sps=_sp.get("score",np.nan)
                _col.metric(_spname,f"{_sps:.0f}/100" if np.isfinite(_sps) else "—",help="Puntuación experimental de control/simetría dentro de este plano; no es porcentaje de normalidad.")
                _col.caption(f"Cobertura {_sp.get('coverage_pct',0):.0f}%")
        if idx["context"]=="frontal":
            st.warning(
                "**Cámara frontal única:** el ICBF (Índice de Control Biomecánico Frontal) se centra en pelvis, tronco, hombros-pelvis y COM (Centro de Masa estimado 2D) / BOS (Base de Sustentación), integrando magnitud postural, balance lateral y consistencia cíclica. Rodilla frontal y pie/retropié NO penalizan el compuesto."
            )
        elif idx["context"]=="frontal+lateral":
            st.success(
                "**Frontal + lateral:** se separan el control biomecánico frontal (pelvis, tronco, hombros y COM) y el control cinemático sagital (cadera, rodilla y tobillo-pie)."
            )
        else:
            st.info("**Lateral única:** el compuesto se limita a simetría sagital bilateral; las variables exclusivas del plano frontal no se puntúan.")

        rows=[]
        for name,weight in idx["domain_weights"].items():
            sc=idx["domains"].get(name,np.nan)
            rows.append({
                "Dominio":name,
                "Peso dentro de esta vista":f"{weight*100:.0f}%",
                "Puntuación":round(sc,1) if np.isfinite(sc) else np.nan,
                "Interpretación":_domain_classification(sc),
                "Datos principales":idx["details"].get(name,""),
            })
        domain_df=pd.DataFrame(rows)
        st.dataframe(domain_df,use_container_width=True,hide_index=True)

        plot_df=domain_df.dropna(subset=["Puntuación"])[["Dominio","Puntuación"]].copy()
        if not plot_df.empty:
            fig,ax=plt.subplots(figsize=(9,4.6))
            y=np.arange(len(plot_df))
            ax.barh(y,plot_df["Puntuación"].to_numpy(dtype=float))
            ax.set_yticks(y); ax.set_yticklabels(plot_df["Dominio"].tolist()); ax.invert_yaxis()
            ax.set_xlim(0,100)
            ax.set_xlabel("Puntuación del dominio (/100; mayor = mayor simetría/control dentro del modelo)")
            ax.set_title(idx["profile_name"])
            ax.axvline(50,linestyle="--",linewidth=1); ax.axvline(70,linestyle="--",linewidth=1); ax.axvline(85,linestyle="--",linewidth=1)
            fig.tight_layout(); st.pyplot(fig,use_container_width=True); plt.close(fig)

        st.markdown("### Variables deliberadamente no penalizadoras")
        desc_rows=[{"Variable / bloque":k,"Tratamiento metodológico":v} for k,v in idx["descriptive"].items()]
        if desc_rows: st.dataframe(pd.DataFrame(desc_rows),use_container_width=True,hide_index=True)
        st.caption("La exclusión de una variable del compuesto no significa que carezca de valor clínico: puede seguir siendo útil como descriptor, para comparar lados o para seguimiento longitudinal estandarizado.")

        st.markdown("### Prioridades dentro del perfil puntuable")
        ranked=idx["ranked_deficits"]
        if ranked:
            deficit_rows=[]
            for rank,(name,sc) in enumerate(ranked,1):
                deficit_rows.append({"Prioridad":rank,"Dominio":name,"Puntuación":round(sc,1),"Discrepancia relativa":round(100.0-sc,1),"Interpretación":_domain_classification(sc)})
            st.dataframe(pd.DataFrame(deficit_rows),use_container_width=True,hide_index=True)
        else:
            st.info("No hay suficientes dominios puntuables para ordenar prioridades biomecánicas en esta vista.")

        with st.expander("Metodología del análisis biomecánico por tracking · Versión 30"):
            st.write(
                "ICBF (Índice de Control Biomecánico Frontal) / ICS (Índice de Control Cinemático Sagital) 0.4 evita un único índice para cualquier plano. En frontal prioriza control postural y transferencia mediolateral: pelvis, tronco, hombros-pelvis y COM (Centro de Masa estimado 2D) / BOS (Base de Sustentación). Cada dominio integra magnitud, balance lateral y repetibilidad cíclica cuando están disponibles. La agregación frontal es no compensatoria para evitar que un dominio conservado oculte varios déficits importantes. Rodilla frontal y pie/retropié no penalizan. Con frontal+lateral se muestra además un perfil sagital independiente de simetría D/I de cadera, rodilla y tobillo-pie. Los umbrales siguen siendo operacionales/experimentales para seguimiento, no rangos diagnósticos universales."
            )
            st.write("Este índice es interno/experimental y está pensado para seguimiento y síntesis; no equivale a porcentaje de normalidad ni a diagnóstico.")


with tabs[10]:
    st.subheader("Anexo · Diccionario completo de variables")
    st.caption("V61 · Diccionario maestro: integra variables 2D frontal/posterior, 2D lateral, biplanares por fase, 3D estimado/no métrico y controles/variables del 3D calibrado cuando estén disponibles.")
    if not st.session_state.get("metrics_done"):
        st.info("Calcula primero los resultados de la sesión para generar el anexo completo.")
    else:
        _annex=_v61_master_annex(st.session_state.get("metrics", []))
        if _annex.empty:
            st.info("No hay variables disponibles.")
        else:
            _groups=["Todos"]+list(dict.fromkeys(_annex["Grupo"].tolist()))
            _af1,_af2=st.columns([1,2])
            with _af1:
                _grp=st.selectbox("Grupo", _groups, key="annex_group")
            with _af2:
                _query=st.text_input("Buscar variable", placeholder="Ej.: COM, pelvis, cadencia, rodilla...", key="annex_search")
            _show=_annex.copy()
            if _grp!="Todos": _show=_show[_show["Grupo"]==_grp]
            if _query.strip():
                _q=_query.strip().lower()
                _mask=_show.astype(str).apply(lambda col: col.str.lower().str.contains(_q, regex=False)).any(axis=1)
                _show=_show[_mask]
            st.dataframe(_show, use_container_width=True, hide_index=True, height=620)
            st.caption(f"Mostrando {len(_show)} de {len(_annex)} variables. Las referencias son contextuales cuando la métrica no dispone de un rango normativo directamente transferible.")
            st.download_button("⬇️ Descargar anexo completo CSV", data=_annex.to_csv(index=False).encode("utf-8-sig"), file_name=f"anexo_variables_{_safe_filename(st.session_state.get('patient_code','paciente'))}.csv", mime="text/csv", use_container_width=True)


st.divider()
st.caption(f"PhysioSentinel Gait Versión {APP_VERSION} · análisis 2D monocámara · biplanar frontal+lateral · 3D estimado/no métrico · 3D calibrado · informe integrado · anexo maestro de variables · exportación")



# === V53 · Interpretación biomecánica biplanar del ciclo ===
# Capa interpretativa no diagnóstica. No modifica sincronización, IC, TO ni segmentación V52.
def _v53_safe_float(x):
    try:
        v = float(x)
        return v if np.isfinite(v) else None
    except Exception:
        return None

def _v53_find_first(mapping, keys):
    if not isinstance(mapping, dict):
        return None
    for k in keys:
        if k in mapping:
            v = _v53_safe_float(mapping.get(k))
            if v is not None:
                return v
    return None

def _v53_quality_note(to_status, quality):
    q = str(quality or "").strip()
    s = str(to_status or "").strip()
    if "jerárqu" in s.lower():
        return ("El TO (Toe Off / Despegue del Pie) procede de confirmación jerárquica: "
                "la vista lateral es primaria y la frontal aporta respaldo secundario. "
                "La interpretación debe conservar esta incertidumbre.")
    if q:
        return f"Calidad heredada del ciclo biplanar: {q}."
    return "La calidad debe interpretarse junto con los controles de sincronización y eventos de las versiones previas."

def _v53_directional_text(name, ic, to, next_ic, stance_rom, swing_rom):
    # Descripción puramente observacional: evita normalidad/patología y evita inferir signos clínicos.
    vals = [v for v in [ic, to, next_ic, stance_rom, swing_rom] if v is not None]
    if not vals:
        return None
    parts = []
    if ic is not None and to is not None:
        d = to - ic
        if abs(d) < 1.0:
            parts.append(f"entre IC y TO cambia {d:+.1f}°")
        else:
            parts.append(f"entre IC y TO presenta un cambio de {d:+.1f}°")
    if stance_rom is not None:
        parts.append(f"ROM (Rango de Movimiento) en apoyo {stance_rom:.1f}°")
    if swing_rom is not None:
        parts.append(f"ROM (Rango de Movimiento) en oscilación {swing_rom:.1f}°")
    if next_ic is not None and ic is not None:
        parts.append(f"cierre IC→IC {next_ic-ic:+.1f}°")
    return f"{name}: " + "; ".join(parts) + "."

def _v53_interpret_from_rows(rows, summary):
    """rows: lista de dicts V52 por variable/fase/evento. Devuelve observaciones sin diagnosticar."""
    out = []
    if not isinstance(rows, list):
        return out
    by_var = {}
    for r in rows:
        if not isinstance(r, dict):
            continue
        name = str(r.get("variable") or r.get("name") or r.get("metrica") or "").strip()
        if not name:
            continue
        by_var.setdefault(name, []).append(r)
    for name, rr in by_var.items():
        flat = {}
        for r in rr:
            for k, v in r.items():
                if k not in flat or flat[k] in (None, ""):
                    flat[k] = v
        ic = _v53_find_first(flat, ["ic_deg","valor_ic","IC","ic"])
        to = _v53_find_first(flat, ["to_deg","valor_to","TO","to"])
        nic = _v53_find_first(flat, ["next_ic_deg","valor_siguiente_ic","next_ic","IC_siguiente"])
        sr = _v53_find_first(flat, ["stance_rom_deg","rom_apoyo","ROM_apoyo","stance_rom"])
        wr = _v53_find_first(flat, ["swing_rom_deg","rom_oscilacion","ROM_oscilacion","swing_rom"])
        txt = _v53_directional_text(name, ic, to, nic, sr, wr)
        if txt:
            out.append({"variable": name, "observacion": txt})
    return out

def _v53_render_panel():
    # Buscar estructuras V52 sin imponer un nombre único.
    ss = st.session_state
    summary = None
    rows = None
    for key in ["v52_summary", "v52_resumen", "biplanar_v52_summary", "v52_cycle_summary"]:
        if key in ss and isinstance(ss[key], dict):
            summary = ss[key]
            break
    for key in ["v52_phase_metrics", "v52_metricas_fase", "biplanar_v52_phase_metrics",
                "v52_variable_phase_metrics", "v52_rows"]:
        if key in ss:
            obj = ss[key]
            if isinstance(obj, pd.DataFrame):
                rows = obj.to_dict("records")
            elif isinstance(obj, list):
                rows = obj
            break

    # Fallback: reconstruir un resumen mínimo desde claves escalares conocidas.
    if summary is None:
        summary = {}
        scalar_aliases = {
            "stance_pct": ["v52_stance_pct","v52_apoyo_pct"],
            "swing_pct": ["v52_swing_pct","v52_oscilacion_pct"],
            "cycle_duration_s": ["v52_cycle_duration_s","v52_duracion_ciclo_s"],
            "to_pct": ["v52_to_pct","v51_selected_to_pct","v51_to_selected_pct"],
            "to_status": ["v51_status","v51_to_status"],
            "quality": ["v52_quality","v51_quality"],
        }
        for dst, aliases in scalar_aliases.items():
            for a in aliases:
                if a in ss:
                    summary[dst] = ss[a]
                    break

    has_any = bool(summary) or bool(rows)
    if not has_any:
        return

    st.markdown("---")
    st.subheader("V53 · Interpretación biomecánica biplanar del ciclo")
    st.caption(
        "Capa interpretativa explicable sobre V52. Describe el patrón observado; "
        "no establece por sí sola normalidad, patología, diagnóstico ni pronóstico."
    )

    stance = _v53_find_first(summary, ["stance_pct","apoyo_pct","v52_stance_pct"])
    swing = _v53_find_first(summary, ["swing_pct","oscilacion_pct","v52_swing_pct"])
    dur = _v53_find_first(summary, ["cycle_duration_s","duracion_ciclo_s"])
    to_pct = _v53_find_first(summary, ["to_pct","selected_to_pct","v51_selected_to_pct"])
    status = summary.get("to_status") or summary.get("status") or ss.get("v51_status")
    quality = summary.get("quality") or ss.get("v52_quality") or ss.get("v51_quality")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Ciclo IC→IC", f"{dur:.2f} s" if dur is not None else "—")
    c2.metric("Apoyo", f"{stance:.1f} %" if stance is not None else "—")
    c3.metric("TO", f"{to_pct:.1f} %" if to_pct is not None else "—")
    c4.metric("Oscilación", f"{swing:.1f} %" if swing is not None else "—")

    st.info(_v53_quality_note(status, quality))

    observations = _v53_interpret_from_rows(rows or [], summary)
    if observations:
        st.markdown("#### Medición observada")
        for item in observations:
            st.write("• " + item["observacion"])
    else:
        st.write(
            "El ciclo está segmentado en IC (Contacto Inicial), apoyo, TO (Despegue del Pie), "
            "oscilación y siguiente IC. Las conclusiones angulares deben leerse desde las "
            "curvas y métricas V52 disponibles para este ciclo."
        )

    st.markdown("#### Integración biomecánica")
    st.write(
        "La vista frontal/posterior aporta información del control en el plano frontal "
        "(pelvis, tronco, rodilla, orientación del pie y retropié), mientras que la vista "
        "lateral aporta la cinemática sagital de cadera, rodilla y tobillo. Ambas se presentan "
        "sobre el mismo ciclo físico 0–100 %, por lo que pueden relacionarse temporalmente "
        "sin convertir el análisis en una reconstrucción 3D (Tridimensional)."
    )
    st.warning(
        "Interpretación clínica: los valores observados no se etiquetan automáticamente como "
        "normales o patológicos. La perspectiva 2D (Bidimensional), la colocación de cámara, "
        "la oclusión y la calidad del TO condicionan la lectura."
    )

    # Persistencia para exportadores que serializan session_state.
    ss["v53_interpretation"] = {
        "version": "53-corregida",
        "stance_pct": stance,
        "swing_pct": swing,
        "to_pct": to_pct,
        "cycle_duration_s": dur,
        "to_status": status,
        "quality": quality,
        "observations": observations,
        "scope": "Interpretación biomecánica biplanar 2D no diagnóstica",
    }

# Render al final para no interferir con las capas V47–V52.
try:
    _v53_render_panel()
except Exception as _v53_exc:
    st.caption(f"V53 · interpretación no disponible en esta ejecución: {_v53_exc}")




# === V54 · Preparación geométrica pre-3D ===
def _v54_geometry_panel():
    st.markdown("---")
    st.subheader("V54 CORREGIDA 3 · Preparación geométrica para futura reconstrucción 3D")
    st.caption(
        "V54 no reconstruye todavía en 3D (Tridimensional). Comprueba qué piezas geométricas "
        "faltan antes de poder calibrar y triangular landmarks de ambas cámaras."
    )

    # Reutiliza únicamente estado ya calculado; no altera V47–V53.
    _ss = st.session_state
    _v48_profile = _ss.get("biplanar_v48_profile")
    _v48_summary = _v48_profile.get("summary") if isinstance(_v48_profile, dict) else {}
    _cycle_available = bool(
        isinstance(_v48_profile, dict)
        and isinstance(_v48_summary, dict)
        and _v48_summary.get("selected_pair_index") is not None
        and str(_v48_summary.get("front_side", "")).strip()
        and str(_v48_summary.get("lateral_side", "")).strip()
    )

    _rows = [
        {"Requisito": "Sincronización física", "Estado": "OK",
         "Detalle": "Capa V47 congelada; V54 no modifica el reloj físico."},
        {"Requisito": "Ciclo físico homólogo", "Estado": "OK" if _cycle_available else "Por verificar",
         "Detalle": (
             f"Par #{int(_v48_summary.get('selected_pair_index'))} · "
             f"{_v48_summary.get('front_side','?')}/{_v48_summary.get('lateral_side','?')} · "
             f"confianza {_v48_summary.get('integration_confidence','—')} · "
             f"ΔIC {_v48_summary.get('ic_alignment_delta_ms', float('nan')):+.1f} ms"
             if _cycle_available else
             "Necesario para relacionar el mismo instante corporal entre cámaras."
         )},
        {"Requisito": "Intrínsecos cam01", "Estado": "Pendiente",
         "Detalle": "Matriz K y coeficientes de distorsión."},
        {"Requisito": "Intrínsecos cam02", "Estado": "Pendiente",
         "Detalle": "Matriz K y coeficientes de distorsión."},
        {"Requisito": "Extrínsecos cam01↔cam02", "Estado": "Pendiente",
         "Detalle": "Rotación R y traslación t entre cámaras."},
        {"Requisito": "Solapamiento simultáneo de campo visual", "Estado": "Por verificar",
         "Detalle": "Los mismos landmarks deben ser visibles simultáneamente para triangular."},
        {"Requisito": "Triangulación 3D", "Estado": "Desactivada",
         "Detalle": "No se habilita hasta disponer de calibración geométrica válida."},
    ]
    _df54 = pd.DataFrame(_rows)
    st.dataframe(_df54, use_container_width=True, hide_index=True)

    st.info(
        "Siguiente etapa: calibración intrínseca y extrínseca. Solo después se podrán triangular "
        "los mismos landmarks HALPE26 observados simultáneamente desde frontal/posterior y lateral."
    )
    st.warning(
        "Dos vídeos sincronizados no producen 3D por sí solos. Sin parámetros de cámara válidos, "
        "las coordenadas 2D de ambas vistas no tienen una intersección espacial físicamente interpretable."
    )

    _payload54 = {
        "version": "54-corregida-3",
        "purpose": "Preparación geométrica pre-3D",
        "sync_layer": "V47 congelada",
        "homologous_cycle_available": _cycle_available,
        "homologous_pair_index": _v48_summary.get("selected_pair_index") if _cycle_available else None,
        "front_side": _v48_summary.get("front_side") if _cycle_available else None,
        "lateral_side": _v48_summary.get("lateral_side") if _cycle_available else None,
        "integration_confidence": _v48_summary.get("integration_confidence") if _cycle_available else None,
        "ic_alignment_delta_ms": _v48_summary.get("ic_alignment_delta_ms") if _cycle_available else None,
        "front_cycle_s": _v48_summary.get("front_cycle_s") if _cycle_available else None,
        "lateral_cycle_s": _v48_summary.get("lateral_cycle_s") if _cycle_available else None,
        "intrinsics_cam01": "pendiente",
        "intrinsics_cam02": "pendiente",
        "extrinsics_cam01_cam02": "pendiente",
        "simultaneous_fov_overlap": "por verificar",
        "triangulation_3d_enabled": False,
        "source": "biplanar_v48_profile leído al renderizar panel",
    }
    _ss["v54_geometry_readiness"] = _payload54

try:
    _v54_geometry_panel()
except Exception as _v54_exc:
    st.caption(f"V54 · preparación geométrica no disponible: {_v54_exc}")

# === V55 · Calibración geométrica de cámaras ===

def _v55_norm_name(value):
    return re.sub(r"[^a-z0-9]+", "", str(value or "").lower())


def _v55_float(value):
    try:
        x = float(str(value).strip().replace(",", "."))
        return x if np.isfinite(x) else None
    except Exception:
        return None


def _v55_flatten_xml(root):
    """Aplana tags/atributos XML para aceptar variantes habituales del modelo Tsai."""
    out = {}
    for elem in root.iter():
        tag = _v55_norm_name(elem.tag.split("}")[-1])
        txt = (elem.text or "").strip()
        if txt and len(list(elem)) == 0:
            fv = _v55_float(txt)
            out[tag] = fv if fv is not None else txt
        for k, v in elem.attrib.items():
            key = _v55_norm_name(k)
            fv = _v55_float(v)
            out[key] = fv if fv is not None else v
            out[f"{tag}{key}"] = fv if fv is not None else v
    return out


def _v55_pick(flat, aliases):
    for a in aliases:
        aa = _v55_norm_name(a)
        if aa in flat:
            return flat[aa]
    # fallback por sufijo, útil si la clave quedó GeometryWidth, IntrinsicFocal, etc.
    for a in aliases:
        aa = _v55_norm_name(a)
        for k, v in flat.items():
            if k.endswith(aa):
                return v
    return None


def _v55_rotation_tsai(rx, ry, rz, unit="radians"):
    """Convención Tsai clásica: world -> camera."""
    if unit == "degrees":
        rx, ry, rz = np.deg2rad([rx, ry, rz])
    sa, ca = np.sin(rx), np.cos(rx)
    sb, cb = np.sin(ry), np.cos(ry)
    sg, cg = np.sin(rz), np.cos(rz)
    return np.array([
        [cb*cg,              cg*sa*sb - ca*sg,  sa*sg + ca*cg*sb],
        [cb*sg,              sa*sb*sg + ca*cg,  ca*sb*sg - cg*sa],
        [-sb,                cb*sa,               ca*cb],
    ], dtype=float)


def _v55_parse_tsai_xml(xml_bytes, angle_unit="radians", source_name=""):
    result = {
        "source_name": str(source_name or ""),
        "valid": False,
        "errors": [],
        "warnings": [],
        "parameters": {},
    }
    try:
        root = ET.fromstring(xml_bytes)
    except Exception as exc:
        result["errors"].append(f"XML no válido: {exc}")
        return result

    flat = _v55_flatten_xml(root)

    p = {
        "width": _v55_pick(flat, ["width", "imagewidth"]),
        "height": _v55_pick(flat, ["height", "imageheight"]),
        "ncx": _v55_pick(flat, ["ncx"]),
        "nfx": _v55_pick(flat, ["nfx"]),
        "dx": _v55_pick(flat, ["dx"]),
        "dy": _v55_pick(flat, ["dy"]),
        "dpx": _v55_pick(flat, ["dpx"]),
        "dpy": _v55_pick(flat, ["dpy"]),
        "focal": _v55_pick(flat, ["focal", "f"]),
        "kappa1": _v55_pick(flat, ["kappa1", "k1"]),
        "cx": _v55_pick(flat, ["cx", "centerx"]),
        "cy": _v55_pick(flat, ["cy", "centery"]),
        "sx": _v55_pick(flat, ["sx"]),
        "tx": _v55_pick(flat, ["tx"]),
        "ty": _v55_pick(flat, ["ty"]),
        "tz": _v55_pick(flat, ["tz"]),
        "rx": _v55_pick(flat, ["rx"]),
        "ry": _v55_pick(flat, ["ry"]),
        "rz": _v55_pick(flat, ["rz"]),
    }

    # Derivaciones estándar de Tsai si dpx/dpy no vienen explícitos.
    if p["dpx"] is None and all(p.get(k) not in (None, 0) for k in ["dx", "ncx", "nfx"]):
        p["dpx"] = float(p["dx"]) * float(p["ncx"]) / float(p["nfx"])
        result["warnings"].append("dpx derivado como dx·ncx/nfx.")
    if p["dpy"] is None and p["dy"] not in (None, 0):
        p["dpy"] = float(p["dy"])
        result["warnings"].append("dpy derivado desde dy.")
    if p["sx"] is None:
        p["sx"] = 1.0
        result["warnings"].append("sx ausente; se usa 1.0.")
    if p["kappa1"] is None:
        p["kappa1"] = 0.0
        result["warnings"].append("kappa1 ausente; se conserva 0.0 solo como fallback.")

    result["parameters"] = p

    required_intrinsic = ["focal", "dpx", "dpy", "cx", "cy", "sx"]
    required_extrinsic = ["tx", "ty", "tz", "rx", "ry", "rz"]
    missing_i = [k for k in required_intrinsic if p.get(k) is None]
    missing_e = [k for k in required_extrinsic if p.get(k) is None]

    if missing_i:
        result["errors"].append("Faltan intrínsecos: " + ", ".join(missing_i))
    if missing_e:
        result["errors"].append("Faltan extrínsecos: " + ", ".join(missing_e))
    if result["errors"]:
        return result

    try:
        f = float(p["focal"])
        dpx = float(p["dpx"])
        dpy = float(p["dpy"])
        sx = float(p["sx"])
        cx = float(p["cx"])
        cy = float(p["cy"])
        if abs(dpx) < 1e-12 or abs(dpy) < 1e-12:
            raise ValueError("dpx/dpy no pueden ser cero.")

        # Matriz intrínseca equivalente del componente pinhole.
        fx = f * sx / dpx
        fy = f / dpy
        K = np.array([
            [fx, 0.0, cx],
            [0.0, fy, cy],
            [0.0, 0.0, 1.0],
        ], dtype=float)

        R = _v55_rotation_tsai(
            float(p["rx"]), float(p["ry"]), float(p["rz"]), unit=angle_unit
        )
        t = np.array([float(p["tx"]), float(p["ty"]), float(p["tz"])], dtype=float).reshape(3, 1)
        Rt = np.hstack([R, t])
        P = K @ Rt
        C = (-R.T @ t).reshape(3)

        detR = float(np.linalg.det(R))
        orth_err = float(np.linalg.norm(R @ R.T - np.eye(3), ord="fro"))

        result.update({
            "valid": bool(
                np.isfinite(K).all()
                and np.isfinite(R).all()
                and np.isfinite(t).all()
                and np.isfinite(P).all()
                and abs(detR - 1.0) < 1e-6
                and orth_err < 1e-6
            ),
            "angle_unit": angle_unit,
            "K": K.tolist(),
            "R": R.tolist(),
            "t": t.reshape(3).tolist(),
            "P": P.tolist(),
            "camera_center_world": C.tolist(),
            "fx_px": float(fx),
            "fy_px": float(fy),
            "det_R": detR,
            "orthogonality_error": orth_err,
            "distortion_model": "Tsai radial 1er orden",
            "kappa1": float(p.get("kappa1") or 0.0),
        })
        if not result["valid"]:
            result["errors"].append("La matriz de rotación no supera el control geométrico.")
    except Exception as exc:
        result["errors"].append(f"No se pudo construir K/R/t/P: {exc}")

    return result


def _v55_relative_geometry(cam1, cam2):
    if not (isinstance(cam1, dict) and isinstance(cam2, dict) and cam1.get("valid") and cam2.get("valid")):
        return {"valid": False}

    R1 = np.asarray(cam1["R"], dtype=float)
    R2 = np.asarray(cam2["R"], dtype=float)
    t1 = np.asarray(cam1["t"], dtype=float).reshape(3, 1)
    t2 = np.asarray(cam2["t"], dtype=float).reshape(3, 1)
    C1 = np.asarray(cam1["camera_center_world"], dtype=float).reshape(3)
    C2 = np.asarray(cam2["camera_center_world"], dtype=float).reshape(3)

    # x2 = R21*x1 + t21, partiendo de x_cam = R*x_world + t
    R21 = R2 @ R1.T
    t21 = t2 - R21 @ t1
    baseline = float(np.linalg.norm(C2 - C1))
    detR = float(np.linalg.det(R21))
    orth_err = float(np.linalg.norm(R21 @ R21.T - np.eye(3), ord="fro"))

    return {
        "valid": bool(np.isfinite(baseline) and baseline > 1e-9 and abs(detR - 1.0) < 1e-6 and orth_err < 1e-6),
        "R_cam01_to_cam02": R21.tolist(),
        "t_cam01_to_cam02": t21.reshape(3).tolist(),
        "baseline_world_units": baseline,
        "det_R_relative": detR,
        "orthogonality_error_relative": orth_err,
    }


def _v55_matrix_df(M):
    a = np.asarray(M, dtype=float)
    return pd.DataFrame(a)


def _v55_render_panel():
    st.markdown("---")
    st.subheader("V55 CORREGIDA · Calibración geométrica de las cámaras")
    st.caption(
        "Importa la calibración Tsai de cam01 y cam02, construye K (intrínsecos), "
        "R/t (extrínsecos) y P = K[R|t]. V55 valida geometría; todavía NO triangula 3D."
    )

    # V55 CORREGIDA · no depender de una copia potencialmente obsoleta de V54.
    # La fuente de verdad del ciclo homólogo es el perfil V48 vigente en session_state.
    _v48_live = st.session_state.get("biplanar_v48_profile") or {}
    _v48_summary_live = (_v48_live.get("summary") or {}) if isinstance(_v48_live, dict) else {}

    cycle_ok = bool(
        _v48_summary_live.get("selected_pair_index") is not None
        and _v48_summary_live.get("front_side")
        and _v48_summary_live.get("lateral_side")
    )

    # Fallback solo para compatibilidad si aún no existe V48 en esta ejecución.
    if not cycle_ok:
        _v54_fallback = st.session_state.get("v54_geometry_readiness") or {}
        cycle_ok = bool(_v54_fallback.get("homologous_cycle_available"))
    else:
        _v54_fallback = {}

    if cycle_ok:
        _pair_idx = (
            _v48_summary_live.get("selected_pair_index")
            if _v48_summary_live else _v54_fallback.get("homologous_pair_index")
        )
        _front_side = (
            _v48_summary_live.get("front_side")
            if _v48_summary_live else _v54_fallback.get("front_side")
        )
        _lat_side = (
            _v48_summary_live.get("lateral_side")
            if _v48_summary_live else _v54_fallback.get("lateral_side")
        )
        _conf = (
            _v48_summary_live.get("integration_confidence")
            if _v48_summary_live else _v54_fallback.get("integration_confidence")
        )
        st.success(
            f"Base temporal lista: ciclo homólogo #{_pair_idx if _pair_idx is not None else '—'} · "
            f"{_front_side or '?'}/{_lat_side or '?'} · confianza {_conf or '—'}."
        )
    else:
        st.warning("V54/V48 aún no confirma ciclo físico homólogo en esta ejecución.")

    st.info(
        "Para el ensayo GPJATK p12s3 que estamos usando: cam01 corresponde a C2 "
        "(frontal/anterior) y cam02 a C3 (lateral). Selecciona los XML de calibración "
        "correspondientes a esas cámaras dentro de la carpeta Calibration."
    )

    angle_label = st.selectbox(
        "Unidad de Rx, Ry y Rz del XML Tsai",
        ["Radianes", "Grados"],
        index=0,
        key="v55_angle_unit_ui",
        help="V55 no intenta adivinar silenciosamente la unidad. Para los XML Tsai habituales se parte de radianes."
    )
    angle_unit = "radians" if angle_label == "Radianes" else "degrees"

    c1, c2 = st.columns(2, gap="large")
    with c1:
        f1 = st.file_uploader(
            "Calibración XML cam01 · frontal/posterior",
            type=["xml"],
            key="v55_xml_cam01"
        )
    with c2:
        f2 = st.file_uploader(
            "Calibración XML cam02 · lateral",
            type=["xml"],
            key="v55_xml_cam02"
        )

    # Parse on every rerun if files exist; Streamlit UploadedFile is seekable via getvalue().
    cam1 = _v55_parse_tsai_xml(f1.getvalue(), angle_unit, f1.name) if f1 is not None else None
    cam2 = _v55_parse_tsai_xml(f2.getvalue(), angle_unit, f2.name) if f2 is not None else None
    rel = _v55_relative_geometry(cam1, cam2) if cam1 and cam2 else {"valid": False}

    if cam1 is not None or cam2 is not None:
        d1, d2 = st.columns(2, gap="large")
        for col, label, cam in [(d1, "cam01", cam1), (d2, "cam02", cam2)]:
            with col:
                st.markdown(f"#### {label}")
                if not isinstance(cam, dict):
                    st.warning("XML no cargado.")
                    continue
                if cam.get("valid"):
                    st.success("Calibración Tsai leída y geométricamente válida.")
                    p = cam.get("parameters") or {}
                    st.write(
                        f"fx={cam.get('fx_px', float('nan')):.3f} px · "
                        f"fy={cam.get('fy_px', float('nan')):.3f} px · "
                        f"cx={float(p.get('cx')):.3f} · cy={float(p.get('cy')):.3f}"
                    )
                    st.write(
                        f"kappa1={float(cam.get('kappa1',0.0)):.6g} · "
                        f"det(R)={cam.get('det_R', float('nan')):.9f}"
                    )
                    with st.expander(f"Matrices {label}", expanded=False):
                        st.caption("K · matriz intrínseca")
                        st.dataframe(_v55_matrix_df(cam["K"]), use_container_width=True, hide_index=True)
                        st.caption("R · rotación mundo→cámara")
                        st.dataframe(_v55_matrix_df(cam["R"]), use_container_width=True, hide_index=True)
                        st.caption("t · traslación mundo→cámara")
                        st.dataframe(_v55_matrix_df(np.asarray(cam["t"]).reshape(3,1)), use_container_width=True, hide_index=True)
                        st.caption("P = K[R|t]")
                        st.dataframe(_v55_matrix_df(cam["P"]), use_container_width=True, hide_index=True)
                else:
                    st.error("Calibración no utilizable.")
                    for err in cam.get("errors") or []:
                        st.write("• " + str(err))
                for warn in cam.get("warnings") or []:
                    st.caption("Aviso: " + str(warn))

    calibration_ok = bool(
        cycle_ok
        and isinstance(cam1, dict) and cam1.get("valid")
        and isinstance(cam2, dict) and cam2.get("valid")
        and isinstance(rel, dict) and rel.get("valid")
    )

    if isinstance(rel, dict) and rel.get("valid"):
        st.markdown("#### Geometría relativa cam01↔cam02")
        st.metric(
            "Baseline entre centros ópticos",
            f"{float(rel.get('baseline_world_units')):.3f} unidades del XML"
        )
        st.caption(
            "La unidad espacial se conserva exactamente como aparece en la calibración original "
            "(habitualmente milímetros en implementaciones Tsai de este tipo)."
        )
        with st.expander("Transformación relativa cam01→cam02", expanded=False):
            st.caption("R cam01→cam02")
            st.dataframe(_v55_matrix_df(rel["R_cam01_to_cam02"]), use_container_width=True, hide_index=True)
            st.caption("t cam01→cam02")
            st.dataframe(
                _v55_matrix_df(np.asarray(rel["t_cam01_to_cam02"]).reshape(3,1)),
                use_container_width=True,
                hide_index=True
            )

    rows = [
        {
            "Requisito": "Ciclo físico homólogo V54",
            "Estado": "OK" if cycle_ok else "Pendiente",
            "Detalle": "Mismo instante corporal entre ambas vistas."
        },
        {
            "Requisito": "Intrínsecos cam01",
            "Estado": "OK" if isinstance(cam1, dict) and cam1.get("valid") else "Pendiente",
            "Detalle": "K + distorsión Tsai."
        },
        {
            "Requisito": "Intrínsecos cam02",
            "Estado": "OK" if isinstance(cam2, dict) and cam2.get("valid") else "Pendiente",
            "Detalle": "K + distorsión Tsai."
        },
        {
            "Requisito": "Extrínsecos cam01 y cam02",
            "Estado": "OK" if isinstance(rel, dict) and rel.get("valid") else "Pendiente",
            "Detalle": "R/t respecto al mismo sistema mundo + transformación relativa."
        },
        {
            "Requisito": "Matrices de proyección P1/P2",
            "Estado": "OK" if isinstance(cam1, dict) and cam1.get("valid") and isinstance(cam2, dict) and cam2.get("valid") else "Pendiente",
            "Detalle": "Preparadas para V56, tras tratamiento correcto de distorsión."
        },
        {
            "Requisito": "Triangulación 3D",
            "Estado": "Preparada para V56" if calibration_ok else "Desactivada",
            "Detalle": "V55 no genera coordenadas X,Y,Z."
        },
    ]
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    summary = {
        "version": "55-corregida",
        "purpose": "Calibración geométrica de cámaras previa a triangulación 3D",
        "camera_model": "Tsai",
        "angle_unit": angle_unit,
        "homologous_cycle_available": cycle_ok,
        "homologous_pair_index": (
            _v48_summary_live.get("selected_pair_index")
            if isinstance(_v48_summary_live, dict) else None
        ),
        "front_side": (
            _v48_summary_live.get("front_side")
            if isinstance(_v48_summary_live, dict) else None
        ),
        "lateral_side": (
            _v48_summary_live.get("lateral_side")
            if isinstance(_v48_summary_live, dict) else None
        ),
        "integration_confidence": (
            _v48_summary_live.get("integration_confidence")
            if isinstance(_v48_summary_live, dict) else None
        ),
        "ic_alignment_delta_ms": (
            _v48_summary_live.get("ic_alignment_delta_ms")
            if isinstance(_v48_summary_live, dict) else None
        ),
        "front_cycle_s": (
            _v48_summary_live.get("front_cycle_s")
            if isinstance(_v48_summary_live, dict) else None
        ),
        "lateral_cycle_s": (
            _v48_summary_live.get("lateral_cycle_s")
            if isinstance(_v48_summary_live, dict) else None
        ),
        "cycle_source": "biplanar_v48_profile leído en vivo por V55 CORREGIDA",
        "cam01_loaded": bool(cam1 is not None),
        "cam02_loaded": bool(cam2 is not None),
        "cam01_valid": bool(isinstance(cam1, dict) and cam1.get("valid")),
        "cam02_valid": bool(isinstance(cam2, dict) and cam2.get("valid")),
        "relative_geometry_valid": bool(isinstance(rel, dict) and rel.get("valid")),
        "baseline_world_units": (
            float(rel.get("baseline_world_units"))
            if isinstance(rel, dict) and rel.get("valid") else None
        ),
        "projection_matrices_ready": bool(
            isinstance(cam1, dict) and cam1.get("valid")
            and isinstance(cam2, dict) and cam2.get("valid")
        ),
        "v56_triangulation_ready": calibration_ok,
        "triangulation_3d_enabled": False,
        "cam01_source": cam1.get("source_name") if isinstance(cam1, dict) else None,
        "cam02_source": cam2.get("source_name") if isinstance(cam2, dict) else None,
        "note": "V55 valida K, R, t y P. La distorsión radial Tsai debe tratarse antes/durante la triangulación V56.",
    }

    st.session_state["v55_calibration"] = {
        "summary": summary,
        "cam01": cam1,
        "cam02": cam2,
        "relative": rel,
    }

    if calibration_ok:
        st.success(
            "V55 completada para esta ejecución: K1/K2, R1/R2, t1/t2 y P1/P2 disponibles. "
            "La siguiente versión puede abordar triangulación 3D con control de distorsión y error de reproyección."
        )
    elif f1 is None or f2 is None:
        st.caption(
            "Para completar V55 carga los dos XML de calibración. No hace falta volver a cargar los vídeos."
        )


try:
    _v55_render_panel()
except Exception as _v55_exc:
    st.caption(f"V55 · calibración geométrica no disponible: {_v55_exc}")

# === V56 · Triangulación 3D HALPE26 ===

def _v56_cam_params(cam):
    if not isinstance(cam, dict) or not cam.get("valid"):
        return None
    p = cam.get("parameters") or {}
    needed = ["focal", "cx", "cy", "sx", "dpx", "dpy", "kappa1"]
    if any(p.get(k) is None for k in needed):
        return None
    return {
        "f": float(p["focal"]),
        "cx": float(p["cx"]),
        "cy": float(p["cy"]),
        "sx": float(p["sx"]),
        "dpx": float(p["dpx"]),
        "dpy": float(p["dpy"]),
        "k1": float(p.get("kappa1") or 0.0),
        "R": np.asarray(cam["R"], dtype=float),
        "t": np.asarray(cam["t"], dtype=float).reshape(3, 1),
        "C": np.asarray(cam["camera_center_world"], dtype=float).reshape(3),
    }


def _v56_tsai_pixel_to_normalized(u, v, cp):
    """
    Tsai: pixel observado (distorsionado) -> coordenada pinhole normalizada.
    Convención usada por el modelo clásico:
        xu = xd * (1 + kappa1 * r_d^2)
        yu = yd * (1 + kappa1 * r_d^2)
    y luego x_n = xu/f, y_n = yu/f.
    """
    if cp is None:
        return np.nan, np.nan
    xd = (float(u) - cp["cx"]) * cp["dpx"] / cp["sx"]
    yd = (float(v) - cp["cy"]) * cp["dpy"]
    r2 = xd * xd + yd * yd
    scale = 1.0 + cp["k1"] * r2
    xu = xd * scale
    yu = yd * scale
    return float(xu / cp["f"]), float(yu / cp["f"])


def _v56_tsai_undistorted_to_distorted_sensor(xu, yu, k1, max_iter=30):
    """
    Invierte xu = xd*(1+k*r_d^2), yu = yd*(1+k*r_d^2)
    manteniendo la dirección radial. Se usa solo para reproyección/QC.
    """
    xu = float(xu); yu = float(yu); k1 = float(k1)
    ru = float(np.hypot(xu, yu))
    if ru < 1e-15 or abs(k1) < 1e-18:
        return xu, yu

    # Resolver ru = rd + k*rd^3 mediante Newton.
    rd = ru
    for _ in range(max_iter):
        fval = rd + k1 * rd**3 - ru
        deriv = 1.0 + 3.0 * k1 * rd**2
        if abs(deriv) < 1e-15:
            break
        step = fval / deriv
        rd_new = rd - step
        if not np.isfinite(rd_new):
            break
        if abs(rd_new - rd) < 1e-12:
            rd = rd_new
            break
        rd = rd_new

    factor = rd / ru if ru > 0 else 1.0
    return xu * factor, yu * factor


def _v56_project_world_to_pixel(X, cp):
    X = np.asarray(X, dtype=float).reshape(3, 1)
    Xc = cp["R"] @ X + cp["t"]
    z = float(Xc[2, 0])
    if not np.isfinite(z) or z <= 1e-12:
        return np.nan, np.nan, z

    xu = cp["f"] * float(Xc[0, 0]) / z
    yu = cp["f"] * float(Xc[1, 0]) / z
    xd, yd = _v56_tsai_undistorted_to_distorted_sensor(xu, yu, cp["k1"])
    u = cp["sx"] * xd / cp["dpx"] + cp["cx"]
    v = yd / cp["dpy"] + cp["cy"]
    return float(u), float(v), z


def _v56_dlt_two_view(x1n, y1n, x2n, y2n, cp1, cp2):
    """Triangulación lineal DLT en coordenadas pinhole normalizadas."""
    P1 = np.hstack([cp1["R"], cp1["t"]])
    P2 = np.hstack([cp2["R"], cp2["t"]])
    A = np.vstack([
        x1n * P1[2, :] - P1[0, :],
        y1n * P1[2, :] - P1[1, :],
        x2n * P2[2, :] - P2[0, :],
        y2n * P2[2, :] - P2[1, :],
    ])
    try:
        _, _, vt = np.linalg.svd(A)
        Xh = vt[-1, :]
        if abs(float(Xh[3])) < 1e-12:
            return None
        X = Xh[:3] / Xh[3]
        return X.astype(float)
    except Exception:
        return None


def _v56_ray_angle_deg(X, C1, C2):
    X = np.asarray(X, dtype=float).reshape(3)
    a = X - np.asarray(C1, dtype=float).reshape(3)
    b = X - np.asarray(C2, dtype=float).reshape(3)
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na <= 1e-12 or nb <= 1e-12:
        return np.nan
    c = float(np.clip(np.dot(a, b) / (na * nb), -1.0, 1.0))
    return float(np.degrees(np.arccos(c)))


def _v56_interp_landmark(df, fps, physical_t, name, sync_offset_s=0.0):
    """
    Interpola una observación 2D a un tiempo físico común.
    Para cam02: t_fisico = frame/fps - sync_offset_s.
    """
    if not isinstance(df, pd.DataFrame) or df.empty:
        return None
    cols = [f"{name}_x", f"{name}_y", f"{name}_score", "frame"]
    if any(c not in df.columns for c in cols):
        return None

    sub = df[cols].copy()
    for c in cols:
        sub[c] = pd.to_numeric(sub[c], errors="coerce")
    sub = sub.dropna()
    if len(sub) < 2:
        return None

    times = sub["frame"].to_numpy(dtype=float) / float(fps) - float(sync_offset_s)
    order = np.argsort(times)
    times = times[order]
    xs = sub[f"{name}_x"].to_numpy(dtype=float)[order]
    ys = sub[f"{name}_y"].to_numpy(dtype=float)[order]
    ss = sub[f"{name}_score"].to_numpy(dtype=float)[order]

    t = float(physical_t)
    if t < times[0] - 1e-9 or t > times[-1] + 1e-9:
        return None

    x = float(np.interp(t, times, xs))
    y = float(np.interp(t, times, ys))
    s = float(np.interp(t, times, ss))

    # Frame flotante de referencia en el sistema original de la cámara.
    frame_float = (t + float(sync_offset_s)) * float(fps)
    return {"x": x, "y": y, "score": s, "frame_float": float(frame_float)}


def _v56_cam2_landmark_name(name, lr_swapped=False):
    if not lr_swapped:
        return name
    if name.startswith("L") and ("R" + name[1:]) in HALPE26:
        return "R" + name[1:]
    if name.startswith("R") and ("L" + name[1:]) in HALPE26:
        return "L" + name[1:]
    return name


def _v56_build_time_grid(v48_summary, df1, fps1, df2, fps2, sync_offset_s):
    if not isinstance(v48_summary, dict):
        return np.array([], dtype=float)

    f0 = v48_summary.get("front_ic_frame")
    f1 = v48_summary.get("front_next_ic_frame")
    if f0 is None or f1 is None:
        return np.array([], dtype=float)

    t0 = float(f0) / float(fps1)
    t1 = float(f1) / float(fps1)
    if t1 <= t0:
        return np.array([], dtype=float)

    # Limitar al rango realmente disponible en ambas cámaras.
    d1_min = float(pd.to_numeric(df1["frame"], errors="coerce").min()) / float(fps1)
    d1_max = float(pd.to_numeric(df1["frame"], errors="coerce").max()) / float(fps1)
    d2_min = float(pd.to_numeric(df2["frame"], errors="coerce").min()) / float(fps2) - float(sync_offset_s)
    d2_max = float(pd.to_numeric(df2["frame"], errors="coerce").max()) / float(fps2) - float(sync_offset_s)
    t0 = max(t0, d1_min, d2_min)
    t1 = min(t1, d1_max, d2_max)
    if t1 <= t0:
        return np.array([], dtype=float)

    target_fps = float(min(float(fps1), float(fps2)))
    n = max(2, int(round((t1 - t0) * target_fps)) + 1)
    return np.linspace(t0, t1, n, dtype=float)


def _v56_triangulate_cycle(df1, fps1, df2, fps2, sync_offset_s, v48_summary,
                           v55, score_min=0.20, reproj_max_px=20.0, min_ray_angle_deg=1.0):
    cam1 = (v55 or {}).get("cam01")
    cam2 = (v55 or {}).get("cam02")
    cp1 = _v56_cam_params(cam1)
    cp2 = _v56_cam_params(cam2)
    if cp1 is None or cp2 is None:
        raise RuntimeError("Calibración V55 incompleta.")

    times = _v56_build_time_grid(v48_summary, df1, fps1, df2, fps2, sync_offset_s)
    if len(times) < 2:
        raise RuntimeError("No existe un intervalo físico común suficiente para el ciclo homólogo.")

    lr_swapped = bool(v48_summary.get("lr_swapped", False))
    rows = []

    for phase_i, t_phys in enumerate(times):
        phase_pct = 100.0 * phase_i / max(1, len(times) - 1)

        for name in HALPE26:
            name2 = _v56_cam2_landmark_name(name, lr_swapped=lr_swapped)
            o1 = _v56_interp_landmark(df1, fps1, t_phys, name, sync_offset_s=0.0)
            o2 = _v56_interp_landmark(df2, fps2, t_phys, name2, sync_offset_s=sync_offset_s)

            row = {
                "physical_time_s": float(t_phys),
                "phase_pct": float(phase_pct),
                "landmark": name,
                "cam02_landmark_used": name2,
                "cam01_frame_float": np.nan if o1 is None else o1["frame_float"],
                "cam02_frame_float": np.nan if o2 is None else o2["frame_float"],
                "cam01_x_px": np.nan if o1 is None else o1["x"],
                "cam01_y_px": np.nan if o1 is None else o1["y"],
                "cam01_score": np.nan if o1 is None else o1["score"],
                "cam02_x_px": np.nan if o2 is None else o2["x"],
                "cam02_y_px": np.nan if o2 is None else o2["y"],
                "cam02_score": np.nan if o2 is None else o2["score"],
                "X": np.nan, "Y": np.nan, "Z": np.nan,
                "depth_cam01": np.nan, "depth_cam02": np.nan,
                "ray_angle_deg": np.nan,
                "reproj_cam01_px": np.nan,
                "reproj_cam02_px": np.nan,
                "reproj_mean_px": np.nan,
                "accepted_3d": False,
                "qc_reason": "",
            }

            if o1 is None or o2 is None:
                row["qc_reason"] = "observación 2D ausente"
                rows.append(row)
                continue

            if min(o1["score"], o2["score"]) < float(score_min):
                row["qc_reason"] = "confianza 2D insuficiente"
                rows.append(row)
                continue

            x1n, y1n = _v56_tsai_pixel_to_normalized(o1["x"], o1["y"], cp1)
            x2n, y2n = _v56_tsai_pixel_to_normalized(o2["x"], o2["y"], cp2)
            if not np.isfinite([x1n, y1n, x2n, y2n]).all():
                row["qc_reason"] = "undistorsión Tsai no válida"
                rows.append(row)
                continue

            X = _v56_dlt_two_view(x1n, y1n, x2n, y2n, cp1, cp2)
            if X is None or not np.isfinite(X).all():
                row["qc_reason"] = "triangulación DLT no válida"
                rows.append(row)
                continue

            u1p, v1p, z1 = _v56_project_world_to_pixel(X, cp1)
            u2p, v2p, z2 = _v56_project_world_to_pixel(X, cp2)
            e1 = float(np.hypot(u1p - o1["x"], v1p - o1["y"])) if np.isfinite([u1p, v1p]).all() else np.nan
            e2 = float(np.hypot(u2p - o2["x"], v2p - o2["y"])) if np.isfinite([u2p, v2p]).all() else np.nan
            em = float(np.nanmean([e1, e2])) if np.isfinite([e1, e2]).any() else np.nan
            angle = _v56_ray_angle_deg(X, cp1["C"], cp2["C"])

            row.update({
                "X": float(X[0]), "Y": float(X[1]), "Z": float(X[2]),
                "depth_cam01": float(z1), "depth_cam02": float(z2),
                "ray_angle_deg": float(angle) if np.isfinite(angle) else np.nan,
                "reproj_cam01_px": e1,
                "reproj_cam02_px": e2,
                "reproj_mean_px": em,
            })

            if not (z1 > 0 and z2 > 0):
                row["qc_reason"] = "punto detrás de una cámara"
            elif not np.isfinite(angle) or angle < float(min_ray_angle_deg):
                row["qc_reason"] = "ángulo de triangulación insuficiente"
            elif not np.isfinite(em) or em > float(reproj_max_px):
                row["qc_reason"] = "error de reproyección alto"
            else:
                row["accepted_3d"] = True
                row["qc_reason"] = "aceptado"

            rows.append(row)

    points = pd.DataFrame(rows)
    if points.empty:
        raise RuntimeError("No se generaron candidatos 3D.")

    frame_qc = (
        points.groupby(["physical_time_s", "phase_pct"], as_index=False)
        .agg(
            landmarks_total=("landmark", "count"),
            landmarks_accepted=("accepted_3d", "sum"),
            reproj_median_px=("reproj_mean_px", "median"),
            ray_angle_median_deg=("ray_angle_deg", "median"),
        )
    )
    frame_qc["accepted_pct"] = 100.0 * frame_qc["landmarks_accepted"] / frame_qc["landmarks_total"].clip(lower=1)

    landmark_qc = (
        points.groupby("landmark", as_index=False)
        .agg(
            samples=("landmark", "count"),
            accepted=("accepted_3d", "sum"),
            median_score_cam01=("cam01_score", "median"),
            median_score_cam02=("cam02_score", "median"),
            median_reproj_px=("reproj_mean_px", "median"),
            p95_reproj_px=("reproj_mean_px", lambda s: float(np.nanpercentile(pd.to_numeric(s, errors="coerce").dropna(), 95)) if pd.to_numeric(s, errors="coerce").notna().any() else np.nan),
            median_ray_angle_deg=("ray_angle_deg", "median"),
        )
    )
    landmark_qc["accepted_pct"] = 100.0 * landmark_qc["accepted"] / landmark_qc["samples"].clip(lower=1)

    valid = points[points["accepted_3d"] == True].copy()
    n_total = int(len(points))
    n_acc = int(len(valid))
    reproj_vals = pd.to_numeric(valid["reproj_mean_px"], errors="coerce").dropna()
    angle_vals = pd.to_numeric(valid["ray_angle_deg"], errors="coerce").dropna()

    summary = {
        "version": "56",
        "purpose": "Triangulación 3D HALPE26 sobre un ciclo físico homólogo",
        "triangulation_3d_enabled": True,
        "method": "DLT de dos vistas sobre coordenadas pinhole normalizadas tras corrección radial Tsai",
        "temporal_alignment": "Reloj físico V47; interpolación 2D al mismo tiempo físico",
        "cycle_source": "biplanar_v48_profile",
        "calibration_source": "v55_calibration",
        "front_ic_frame": v48_summary.get("front_ic_frame"),
        "front_next_ic_frame": v48_summary.get("front_next_ic_frame"),
        "sync_offset_cam02_s": float(sync_offset_s),
        "target_time_samples": int(len(times)),
        "halpe26_landmarks": int(len(HALPE26)),
        "candidate_points_3d": n_total,
        "accepted_points_3d": n_acc,
        "accepted_points_pct": float(100.0 * n_acc / max(1, n_total)),
        "median_reprojection_error_px": float(reproj_vals.median()) if len(reproj_vals) else np.nan,
        "p95_reprojection_error_px": float(np.nanpercentile(reproj_vals, 95)) if len(reproj_vals) else np.nan,
        "median_triangulation_angle_deg": float(angle_vals.median()) if len(angle_vals) else np.nan,
        "score_min": float(score_min),
        "reprojection_max_px": float(reproj_max_px),
        "min_ray_angle_deg": float(min_ray_angle_deg),
        "coordinate_unit": "unidad espacial de la calibración XML",
        "lr_swapped_cam02": bool(lr_swapped),
        "quality_note": "Los umbrales V56 son controles internos de calidad; no constituyen umbrales clínicos validados.",
    }
    return {
        "summary": summary,
        "points3d": points,
        "frame_qc": frame_qc,
        "landmark_qc": landmark_qc,
    }


def _v56_skeleton_pairs():
    return [
        ("Head", "Neck"),
        ("Neck", "LShoulder"), ("Neck", "RShoulder"),
        ("LShoulder", "LElbow"), ("LElbow", "LWrist"),
        ("RShoulder", "RElbow"), ("RElbow", "RWrist"),
        ("LShoulder", "LHip"), ("RShoulder", "RHip"),
        ("LHip", "RHip"),
        ("LHip", "LKnee"), ("LKnee", "LAnkle"),
        ("RHip", "RKnee"), ("RKnee", "RAnkle"),
        ("LAnkle", "LHeel"), ("LAnkle", "LBigToe"), ("LBigToe", "LSmallToe"),
        ("RAnkle", "RHeel"), ("RAnkle", "RBigToe"), ("RBigToe", "RSmallToe"),
    ]


def _v56_static_3d_figure(points3d):
    if not isinstance(points3d, pd.DataFrame) or points3d.empty:
        return None
    accepted = points3d[points3d["accepted_3d"] == True].copy()
    if accepted.empty:
        return None

    # Elegir el frame con más landmarks aceptados.
    counts = accepted.groupby("phase_pct")["landmark"].count()
    if counts.empty:
        return None
    phase = float(counts.idxmax())
    snap = accepted[np.isclose(accepted["phase_pct"].astype(float), phase)].copy()
    if len(snap) < 5:
        return None

    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(6.5, 6))
    ax = fig.add_subplot(111, projection="3d")

    by_name = {r["landmark"]: np.array([r["X"], r["Y"], r["Z"]], dtype=float) for _, r in snap.iterrows()}
    for a, b in _v56_skeleton_pairs():
        if a in by_name and b in by_name:
            pa, pb = by_name[a], by_name[b]
            ax.plot([pa[0], pb[0]], [pa[1], pb[1]], [pa[2], pb[2]], linewidth=1.6)

    xyz = np.vstack(list(by_name.values()))
    ax.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], s=18)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title(f"V56 · esqueleto 3D estático · fase {phase:.1f}%")
    try:
        mins = np.nanmin(xyz, axis=0)
        maxs = np.nanmax(xyz, axis=0)
        center = (mins + maxs) / 2.0
        span = float(np.nanmax(maxs - mins))
        span = max(span, 1.0)
        ax.set_xlim(center[0] - span/2, center[0] + span/2)
        ax.set_ylim(center[1] - span/2, center[1] + span/2)
        ax.set_zlim(center[2] - span/2, center[2] + span/2)
    except Exception:
        pass
    fig.tight_layout()
    return fig


def _v56_render_panel():
    if not str(st.session_state.get("analysis_level", "")).startswith("Nivel 3"):
        return
    st.markdown("---")
    st.subheader("V56 CORREGIDA 3 · Triangulación 3D de HALPE26")
    st.caption(
        "Reconstruye X, Y, Z a partir de las dos vistas sincronizadas. "
        "V56 usa el reloj físico V47, el ciclo homólogo V48 y la calibración Tsai V55. "
        "No calcula todavía cinemática articular 3D."
    )

    v55 = st.session_state.get("v55_calibration") or {}
    v55s = (v55.get("summary") or {}) if isinstance(v55, dict) else {}
    v48 = st.session_state.get("biplanar_v48_profile") or {}
    v48s = (v48.get("summary") or {}) if isinstance(v48, dict) else {}
    # V56 CORREGIDA · fuente HALPE26 robusta.
    # Tras calcular resultados, analysis_df/analysis_df2 se eliminan de memoria,
    # pero el pipeline conserva los segmentos 2D biplanares ligeros realmente usados.
    _df1_live = st.session_state.get("analysis_df")
    _df2_live = st.session_state.get("analysis_df2")
    _meta1 = st.session_state.get("meta1") or {}
    _meta2 = st.session_state.get("meta2") or {}

    _df1_saved = st.session_state.get("biplanar_seg_front")
    _df2_saved = st.session_state.get("biplanar_seg_lateral")
    _fps1_saved = float(st.session_state.get("biplanar_fps_front", 0.0) or 0.0)
    _fps2_saved = float(st.session_state.get("biplanar_fps_lateral", 0.0) or 0.0)

    _live_ok = bool(
        isinstance(_df1_live, pd.DataFrame) and not _df1_live.empty
        and isinstance(_df2_live, pd.DataFrame) and not _df2_live.empty
        and float(_meta1.get("fps", 0) or 0) > 0
        and float(_meta2.get("fps", 0) or 0) > 0
    )
    _saved_ok = bool(
        isinstance(_df1_saved, pd.DataFrame) and not _df1_saved.empty
        and isinstance(_df2_saved, pd.DataFrame) and not _df2_saved.empty
        and _fps1_saved > 0 and _fps2_saved > 0
    )

    if _live_ok:
        df1, df2 = _df1_live, _df2_live
        fps1 = float(_meta1["fps"])
        fps2 = float(_meta2["fps"])
        _halpe_source = "analysis_df/analysis_df2 en memoria"
    elif _saved_ok:
        df1, df2 = _df1_saved, _df2_saved
        fps1 = _fps1_saved
        fps2 = _fps2_saved
        _halpe_source = "biplanar_seg_front/biplanar_seg_lateral preservados tras el cálculo"
    else:
        df1 = df2 = None
        fps1 = fps2 = 0.0
        _halpe_source = "no disponible"

    cycle_ok = bool(
        isinstance(v48s, dict)
        and v48s.get("selected_pair_index") is not None
        and v48s.get("front_ic_frame") is not None
        and v48s.get("front_next_ic_frame") is not None
    )
    calib_ok = bool(
        isinstance(v55, dict)
        and isinstance(v55.get("cam01"), dict) and v55["cam01"].get("valid")
        and isinstance(v55.get("cam02"), dict) and v55["cam02"].get("valid")
        and isinstance(v55.get("relative"), dict) and v55["relative"].get("valid")
    )
    data_ok = bool(
        isinstance(df1, pd.DataFrame) and not df1.empty
        and isinstance(df2, pd.DataFrame) and not df2.empty
        and fps1 > 0 and fps2 > 0
    )

    readiness = pd.DataFrame([
        {"Requisito": "Ciclo físico homólogo V48", "Estado": "OK" if cycle_ok else "Pendiente"},
        {"Requisito": "Calibración Tsai V55", "Estado": "OK" if calib_ok else "Pendiente"},
        {"Requisito": "HALPE26 cam01 + cam02 disponibles", "Estado": "OK" if data_ok else "Pendiente"},
        {"Requisito": "Reloj físico maestro V47/V38", "Estado": "OK" if (
            st.session_state.get("biplanar_v46_physical_clock_offset_s") is not None
            and np.isfinite(float(st.session_state.get("biplanar_v46_physical_clock_offset_s")))
        ) else "Pendiente"},
    ])
    st.dataframe(readiness, use_container_width=True, hide_index=True)
    st.caption(f"Fuente HALPE26 usada por V56: **{_halpe_source}**.")

    if not (cycle_ok and calib_ok and data_ok):
        st.warning(
            "V56 necesita que el análisis biplanar y V55 estén completos en esta misma ejecución. "
            "Cuando los cuatro requisitos estén en OK aparecerá el botón de triangulación."
        )
        return

    # V56 CORREGIDA 2 · para triangulación sólo se admite el RELOJ FÍSICO.
    # biplanar_sync_offset_s puede contener el offset de FASE biomecánica V39
    # y queda explícitamente prohibido como fuente temporal 3D.
    _physical_clock_raw = st.session_state.get("biplanar_v46_physical_clock_offset_s")
    if _physical_clock_raw is None or not np.isfinite(float(_physical_clock_raw)):
        st.error(
            "No hay un reloj físico maestro válido. V56 CORREGIDA 2 no utilizará "
            "el desfase de fase biomecánica como sustituto."
        )
        return

    sync_offset = float(_physical_clock_raw)
    _phase_offset = st.session_state.get("biplanar_v46_biomechanical_phase_offset_s")
    try:
        _phase_offset = float(_phase_offset)
    except Exception:
        _phase_offset = np.nan

    st.success(
        f"Listo para triangular: par #{v48s.get('selected_pair_index')} · "
        f"{v48s.get('front_side','?')}/{v48s.get('lateral_side','?')} · "
        f"{fps1:g}/{fps2:g} FPS · reloj físico {sync_offset:+.4f} s."
    )
    if np.isfinite(_phase_offset):
        st.caption(
            f"Desfase de fase biomecánica (solo diagnóstico, NO usado para 3D): "
            f"{_phase_offset:+.4f} s · diferencia fase−reloj físico "
            f"{1000.0*(_phase_offset-sync_offset):+.1f} ms."
        )

    c1, c2, c3 = st.columns(3)
    with c1:
        score_min = st.slider(
            "Confianza 2D mínima",
            min_value=0.05, max_value=0.80, value=0.20, step=0.05,
            key="v56_score_min"
        )
    with c2:
        reproj_max = st.slider(
            "Error reproyección máximo (px)",
            min_value=3.0, max_value=40.0, value=20.0, step=1.0,
            key="v56_reproj_max"
        )
    with c3:
        ray_min = st.slider(
            "Ángulo mínimo entre rayos (°)",
            min_value=0.2, max_value=10.0, value=1.0, step=0.2,
            key="v56_ray_angle_min"
        )

    st.caption(
        "Estos tres valores son controles internos experimentales de calidad. "
        "Los puntos rechazados se conservan en el CSV para auditoría."
    )

    if st.button("🧊 Triangular ciclo homólogo en 3D", type="primary", use_container_width=True):
        try:
            with st.spinner("Triangulando HALPE26, aplicando distorsión Tsai y calculando error de reproyección..."):
                out = _v56_triangulate_cycle(
                    df1, fps1, df2, fps2, sync_offset, v48s, v55,
                    score_min=score_min,
                    reproj_max_px=reproj_max,
                    min_ray_angle_deg=ray_min,
                )
                if isinstance(out, dict) and isinstance(out.get("summary"), dict):
                    out["summary"]["version"] = "56-corregida-3"
                    out["summary"]["halpe26_source"] = _halpe_source
                    out["summary"]["fps_cam01"] = float(fps1)
                    out["summary"]["fps_cam02"] = float(fps2)
                    out["summary"]["sync_offset_source"] = "biplanar_v46_physical_clock_offset_s"
                    out["summary"]["physical_clock_offset_s"] = float(sync_offset)
                    out["summary"]["biomechanical_phase_offset_s"] = (
                        float(_phase_offset) if np.isfinite(_phase_offset) else None
                    )
                    out["summary"]["phase_minus_physical_ms"] = (
                        1000.0 * (float(_phase_offset) - float(sync_offset))
                        if np.isfinite(_phase_offset) else None
                    )
                    out["summary"]["phase_offset_used_for_triangulation"] = False

                st.session_state["v56_triangulation"] = out

                # Payload serializado estable para que la exportación global no
                # dependa de que los DataFrames sigan vivos en un rerun posterior.
                if isinstance(out, dict):
                    _payload = {
                        "summary_json": json.dumps(
                            out.get("summary") or {},
                            ensure_ascii=False, indent=2, default=str
                        ),
                        "summary_csv": pd.DataFrame(
                            [out.get("summary") or {}]
                        ).to_csv(index=False),
                        "points3d_csv": (
                            out.get("points3d").to_csv(index=False)
                            if isinstance(out.get("points3d"), pd.DataFrame) else ""
                        ),
                        "frame_qc_csv": (
                            out.get("frame_qc").to_csv(index=False)
                            if isinstance(out.get("frame_qc"), pd.DataFrame) else ""
                        ),
                        "landmark_qc_csv": (
                            out.get("landmark_qc").to_csv(index=False)
                            if isinstance(out.get("landmark_qc"), pd.DataFrame) else ""
                        ),
                    }
                    _payload["payload_version"] = "56-corregida-3"
                    _payload["source"] = "generado en el momento de triangular"
                    st.session_state["v56_export_payload"] = _payload
            st.success("Triangulación V56 completada.")
        except Exception as exc:
            st.error(f"V56 no pudo completar la triangulación: {exc}")
            with st.expander("Detalles técnicos"):
                st.code(traceback.format_exc())

    out = st.session_state.get("v56_triangulation")
    if not isinstance(out, dict):
        return

    s = out.get("summary") or {}
    pts = out.get("points3d")
    lqc = out.get("landmark_qc")

    a, b, c, d = st.columns(4)
    a.metric("Puntos 3D aceptados", f"{int(s.get('accepted_points_3d',0))}/{int(s.get('candidate_points_3d',0))}")
    b.metric("Aceptación", f"{float(s.get('accepted_points_pct',np.nan)):.1f}%")
    c.metric("Error reproyección mediano", f"{float(s.get('median_reprojection_error_px',np.nan)):.2f} px")
    d.metric("Ángulo mediano de triangulación", f"{float(s.get('median_triangulation_angle_deg',np.nan)):.1f}°")

    p95 = float(s.get("p95_reprojection_error_px", np.nan))
    if np.isfinite(p95):
        st.caption(
            f"P95 del error de reproyección: {p95:.2f} px · "
            "coordenadas X/Y/Z expresadas en la unidad espacial de la calibración XML."
        )

    fig = _v56_static_3d_figure(pts)
    if fig is not None:
        st.pyplot(fig, use_container_width=False)
        try:
            import matplotlib.pyplot as plt
            plt.close(fig)
        except Exception:
            pass

    if isinstance(lqc, pd.DataFrame) and not lqc.empty:
        st.markdown("#### Control de calidad por landmark")
        show = lqc.copy()
        for col in ["accepted_pct", "median_reproj_px", "p95_reproj_px", "median_ray_angle_deg"]:
            if col in show.columns:
                show[col] = pd.to_numeric(show[col], errors="coerce").round(2)
        st.dataframe(show, use_container_width=True, hide_index=True)

    st.info(
        "V56 genera coordenadas 3D y sus controles geométricos. "
        "Todavía no interpreta estos puntos como ángulos articulares 3D ni como cinética. "
        "La siguiente etapa será validar primero la reconstrucción antes de construir la animación/cinemática."
    )

    # Descarga específica de respaldo: permite auditar V56 aunque el usuario
    # todavía no haya generado el ZIP global de la aplicación.
    try:
        _v56_buf = io.BytesIO()
        with zipfile.ZipFile(_v56_buf, "w", zipfile.ZIP_DEFLATED) as _zf:
            _zf.writestr(
                "resumen_triangulacion_v56.json",
                json.dumps(s, ensure_ascii=False, indent=2, default=str).encode("utf-8")
            )
            if isinstance(pts, pd.DataFrame) and not pts.empty:
                _zf.writestr("puntos_3d_halpe26_v56.csv", pts.to_csv(index=False).encode("utf-8-sig"))
            _fq = out.get("frame_qc")
            if isinstance(_fq, pd.DataFrame) and not _fq.empty:
                _zf.writestr("control_calidad_por_frame_v56.csv", _fq.to_csv(index=False).encode("utf-8-sig"))
            if isinstance(lqc, pd.DataFrame) and not lqc.empty:
                _zf.writestr("control_calidad_por_landmark_v56.csv", lqc.to_csv(index=False).encode("utf-8-sig"))
        st.download_button(
            "⬇️ Descargar respaldo V56 (X/Y/Z + QC)",
            data=_v56_buf.getvalue(),
            file_name="PhysioSentinel_V56_triangulacion_3D.zip",
            mime="application/zip",
            use_container_width=True,
        )
    except Exception as _backup_exc:
        st.caption(f"Respaldo V56 no disponible: {_backup_exc}")


try:
    _v56_render_panel()
except Exception as _v56_exc:
    st.caption(f"V56 · triangulación 3D no disponible: {_v56_exc}")


# === V57 · Esqueleto 3D dinámico e interactivo ===

def _v57_camera_presets():
    """
    Cámaras Plotly. Las etiquetas son operativas: el sistema de ejes conserva
    las coordenadas de la calibración XML y no fuerza una redefinición anatómica.
    """
    return {
        "3D oblicua": dict(eye=dict(x=1.45, y=1.45, z=1.15), up=dict(x=0, y=0, z=1)),
        "Frontal": dict(eye=dict(x=0.0, y=2.4, z=0.15), up=dict(x=0, y=0, z=1)),
        "Posterior": dict(eye=dict(x=0.0, y=-2.4, z=0.15), up=dict(x=0, y=0, z=1)),
        "Lateral A": dict(eye=dict(x=2.4, y=0.0, z=0.15), up=dict(x=0, y=0, z=1)),
        "Lateral B": dict(eye=dict(x=-2.4, y=0.0, z=0.15), up=dict(x=0, y=0, z=1)),
        "Superior": dict(eye=dict(x=0.01, y=0.01, z=2.7), up=dict(x=0, y=1, z=0)),
    }


def _v62c_estimated_camera_presets():
    """
    V62 CORREGIDA · cámaras del 3D estimado/no métrico.

    Convención V60 estimada:
      X = horizontal de la cámara frontal
      Y = vertical corporal (altura fusionada frontal/lateral)
      Z = horizontal de la cámara lateral / profundidad aproximada

    Por tanto, la vista frontal debe mirar a lo largo de Z, la lateral a lo
    largo de X y la superior a lo largo de Y. Esta convención es distinta de
    la triangulación calibrada V56/V57 y no modifica sus presets.
    """
    return {
        "3D oblicua": dict(eye=dict(x=1.45, y=1.15, z=1.45), up=dict(x=0, y=1, z=0)),
        "Frontal": dict(eye=dict(x=0.0, y=0.10, z=2.4), up=dict(x=0, y=1, z=0)),
        "Posterior": dict(eye=dict(x=0.0, y=0.10, z=-2.4), up=dict(x=0, y=1, z=0)),
        "Lateral A": dict(eye=dict(x=2.4, y=0.10, z=0.0), up=dict(x=0, y=1, z=0)),
        "Lateral B": dict(eye=dict(x=-2.4, y=0.10, z=0.0), up=dict(x=0, y=1, z=0)),
        "Superior": dict(eye=dict(x=0.0, y=2.7, z=0.01), up=dict(x=0, y=0, z=1)),
    }


def _v57_equal_ranges(points3d):
    """Rangos cúbicos estables para evitar cambios de zoom durante la animación."""
    if not isinstance(points3d, pd.DataFrame) or points3d.empty:
        return None
    d = points3d.copy()
    if "accepted_3d" in d.columns:
        d = d[d["accepted_3d"] == True].copy()
    for c in ["X", "Y", "Z"]:
        d[c] = pd.to_numeric(d[c], errors="coerce")
    d = d.dropna(subset=["X", "Y", "Z"])
    if d.empty:
        return None

    mins = d[["X", "Y", "Z"]].min().to_numpy(dtype=float)
    maxs = d[["X", "Y", "Z"]].max().to_numpy(dtype=float)
    ctr = (mins + maxs) / 2.0
    span = float(np.max(maxs - mins))
    if not np.isfinite(span) or span <= 0:
        span = 1.0
    span *= 1.08

    return {
        "x": [float(ctr[0] - span/2), float(ctr[0] + span/2)],
        "y": [float(ctr[1] - span/2), float(ctr[1] + span/2)],
        "z": [float(ctr[2] - span/2), float(ctr[2] + span/2)],
    }


def _v57_snapshot(points3d, phase):
    d = points3d.copy()
    d["phase_pct"] = pd.to_numeric(d["phase_pct"], errors="coerce")
    d = d[np.isclose(d["phase_pct"], float(phase))].copy()
    if "accepted_3d" in d.columns:
        d = d[d["accepted_3d"] == True].copy()
    for c in ["X", "Y", "Z"]:
        d[c] = pd.to_numeric(d[c], errors="coerce")
    d = d.dropna(subset=["X", "Y", "Z"])
    return d


def _v57_skeleton_trace(snapshot, go, show_labels=False):
    by_name = {
        str(r["landmark"]): (float(r["X"]), float(r["Y"]), float(r["Z"]))
        for _, r in snapshot.iterrows()
    }

    xs, ys, zs = [], [], []
    for a, b in _v56_skeleton_pairs():
        if a in by_name and b in by_name:
            pa, pb = by_name[a], by_name[b]
            xs += [pa[0], pb[0], None]
            ys += [pa[1], pb[1], None]
            zs += [pa[2], pb[2], None]

    return go.Scatter3d(
        x=xs, y=ys, z=zs,
        mode="lines",
        line=dict(width=6),
        hoverinfo="skip",
        name="Segmentos",
        showlegend=False,
    )


def _v57_joint_trace(snapshot, go, show_labels=False):
    custom = []
    for _, r in snapshot.iterrows():
        custom.append([
            str(r["landmark"]),
            float(r.get("phase_pct", np.nan)),
            float(r.get("reproj_mean_px", np.nan)) if pd.notna(r.get("reproj_mean_px", np.nan)) else np.nan,
            float(r.get("ray_angle_deg", np.nan)) if pd.notna(r.get("ray_angle_deg", np.nan)) else np.nan,
        ])

    text = snapshot["landmark"].astype(str).tolist() if show_labels else None
    mode = "markers+text" if show_labels else "markers"

    return go.Scatter3d(
        x=snapshot["X"], y=snapshot["Y"], z=snapshot["Z"],
        mode=mode,
        marker=dict(size=5),
        text=text,
        textposition="top center",
        customdata=custom,
        hovertemplate=(
            "<b>%{customdata[0]}</b><br>"
            "Fase: %{customdata[1]:.1f}%<br>"
            "X: %{x:.2f}<br>Y: %{y:.2f}<br>Z: %{z:.2f}<br>"
            "Error reproyección: %{customdata[2]:.2f} px<br>"
            "Ángulo de rayos: %{customdata[3]:.1f}°"
            "<extra></extra>"
        ),
        name="Landmarks HALPE26",
        showlegend=False,
    )


def _v57_trajectory_trace(points3d, landmark, go):
    d = points3d.copy()
    if "accepted_3d" in d.columns:
        d = d[d["accepted_3d"] == True].copy()
    d = d[d["landmark"].astype(str) == str(landmark)].copy()
    for c in ["phase_pct", "X", "Y", "Z"]:
        d[c] = pd.to_numeric(d[c], errors="coerce")
    d = d.dropna(subset=["phase_pct", "X", "Y", "Z"]).sort_values("phase_pct")
    if d.empty:
        return None

    return go.Scatter3d(
        x=d["X"], y=d["Y"], z=d["Z"],
        mode="lines",
        line=dict(width=3, dash="dot"),
        hovertemplate=(
            f"<b>Trayectoria {landmark}</b><br>"
            "X: %{x:.2f}<br>Y: %{y:.2f}<br>Z: %{z:.2f}"
            "<extra></extra>"
        ),
        name=f"Trayectoria {landmark}",
        showlegend=False,
    )


def _v57_build_figure(points3d, show_labels=False, trajectory_landmark="Ninguna",
                      initial_view="3D oblicua", camera_presets=None):
    import plotly.graph_objects as go

    if not isinstance(points3d, pd.DataFrame) or points3d.empty:
        return None

    d = points3d.copy()
    if "accepted_3d" in d.columns:
        d = d[d["accepted_3d"] == True].copy()

    d["phase_pct"] = pd.to_numeric(d["phase_pct"], errors="coerce")
    phases = sorted([float(x) for x in d["phase_pct"].dropna().unique()])
    if not phases:
        return None

    ranges = _v57_equal_ranges(d)
    if ranges is None:
        return None

    cameras = camera_presets or _v57_camera_presets()
    camera0 = cameras.get(initial_view, cameras["3D oblicua"])

    first = _v57_snapshot(d, phases[0])
    base_traces = [
        _v57_skeleton_trace(first, go, show_labels=show_labels),
        _v57_joint_trace(first, go, show_labels=show_labels),
    ]

    trajectory_trace = None
    if trajectory_landmark and trajectory_landmark != "Ninguna":
        trajectory_trace = _v57_trajectory_trace(d, trajectory_landmark, go)
        if trajectory_trace is not None:
            base_traces.append(trajectory_trace)

    frames = []
    for ph in phases:
        snap = _v57_snapshot(d, ph)
        frame_data = [
            _v57_skeleton_trace(snap, go, show_labels=show_labels),
            _v57_joint_trace(snap, go, show_labels=show_labels),
        ]
        # La trayectoria es estática; se repite para mantener índices de traza estables.
        if trajectory_trace is not None:
            frame_data.append(trajectory_trace)

        frames.append(
            go.Frame(
                data=frame_data,
                name=f"{ph:.4f}",
                traces=list(range(len(frame_data))),
                layout=go.Layout(
                    annotations=[
                        dict(
                            text=f"Fase del ciclo: {ph:.1f}%",
                            x=0.02, y=0.98,
                            xref="paper", yref="paper",
                            showarrow=False,
                            xanchor="left", yanchor="top",
                        )
                    ]
                ),
            )
        )

    # Duraciones: ciclo original ~25 FPS; velocidad cambia el intervalo visual.
    # El usuario puede pausar y rotar libremente en cualquier fase.
    speed_buttons = []
    for label, multiplier in [("0,25×", 0.25), ("0,5×", 0.5), ("1×", 1.0), ("2×", 2.0)]:
        duration = max(1, int(round(40.0 / multiplier)))
        speed_buttons.append(
            dict(
                label=label,
                method="animate",
                args=[
                    None,
                    {
                        "frame": {"duration": duration, "redraw": True},
                        "transition": {"duration": 0},
                        "fromcurrent": True,
                        "mode": "immediate",
                    },
                ],
            )
        )

    sliders = [{
        "active": 0,
        "currentvalue": {"prefix": "Ciclo: ", "suffix": " %"},
        "pad": {"t": 42},
        "len": 0.92,
        "x": 0.04,
        "steps": [
            {
                "label": f"{ph:.0f}",
                "method": "animate",
                "args": [
                    [f"{ph:.4f}"],
                    {
                        "frame": {"duration": 0, "redraw": True},
                        "transition": {"duration": 0},
                        "mode": "immediate",
                    },
                ],
            }
            for ph in phases
        ],
    }]

    camera_buttons = [
        dict(
            label=name,
            method="relayout",
            args=[{"scene.camera": cam}],
        )
        for name, cam in cameras.items()
    ]

    fig = go.Figure(data=base_traces, frames=frames)
    fig.update_layout(
        height=760,
        margin=dict(l=0, r=0, t=55, b=0),
        title=dict(text="V57 · Esqueleto 3D dinámico e interactivo", x=0.02),
        scene=dict(
            xaxis=dict(title="X", range=ranges["x"], autorange=False),
            yaxis=dict(title="Y", range=ranges["y"], autorange=False),
            zaxis=dict(title="Z", range=ranges["z"], autorange=False),
            aspectmode="cube",
            camera=camera0,
        ),
        uirevision="physiosentinel-v57-camera",
        sliders=sliders,
        updatemenus=[
            dict(
                type="buttons",
                direction="left",
                x=0.02, y=1.08,
                xanchor="left", yanchor="top",
                buttons=[
                    dict(
                        label="▶ Reproducir",
                        method="animate",
                        args=[
                            None,
                            {
                                "frame": {"duration": 40, "redraw": True},
                                "transition": {"duration": 0},
                                "fromcurrent": True,
                                "mode": "immediate",
                            },
                        ],
                    ),
                    dict(
                        label="⏸ Pausa",
                        method="animate",
                        args=[
                            [None],
                            {
                                "frame": {"duration": 0, "redraw": False},
                                "transition": {"duration": 0},
                                "mode": "immediate",
                            },
                        ],
                    ),
                ],
            ),
            dict(
                type="dropdown",
                direction="down",
                x=0.40, y=1.08,
                xanchor="left", yanchor="top",
                buttons=speed_buttons,
                showactive=True,
                active=2,
            ),
            dict(
                type="dropdown",
                direction="down",
                x=0.62, y=1.08,
                xanchor="left", yanchor="top",
                buttons=camera_buttons,
                showactive=False,
            ),
        ],
        annotations=[
            dict(
                text=f"Fase del ciclo: {phases[0]:.1f}%",
                x=0.02, y=0.98,
                xref="paper", yref="paper",
                showarrow=False,
                xanchor="left", yanchor="top",
            )
        ],
    )
    return fig


def _v57_render_panel():
    if not str(st.session_state.get("analysis_level", "")).startswith("Nivel 3"):
        return
    st.markdown("---")
    st.subheader("V57 · Esqueleto 3D dinámico e interactivo")
    st.caption(
        "V57 no recalcula la triangulación. Reproduce directamente las coordenadas X/Y/Z "
        "aceptadas por V56 CORREGIDA 3, manteniendo congelados reloj físico, calibración Tsai, DLT y QC."
    )

    out = st.session_state.get("v56_triangulation")
    if not isinstance(out, dict):
        st.info("Primero ejecuta la triangulación V56. En cuanto existan puntos X/Y/Z aparecerá el visor V57.")
        return

    pts = out.get("points3d")
    summary = out.get("summary") or {}
    if not isinstance(pts, pd.DataFrame) or pts.empty:
        st.info("V57 todavía no dispone de puntos 3D de V56.")
        return

    accepted = pts.copy()
    if "accepted_3d" in accepted.columns:
        accepted = accepted[accepted["accepted_3d"] == True].copy()
    if accepted.empty:
        st.warning("No hay puntos 3D aceptados disponibles para animar.")
        return

    phases = pd.to_numeric(accepted["phase_pct"], errors="coerce").dropna().unique()
    phases = sorted([float(x) for x in phases])
    landmark_options = sorted(accepted["landmark"].dropna().astype(str).unique().tolist())

    q1, q2, q3 = st.columns([1.1, 1.2, 1.4])
    with q1:
        initial_view = st.selectbox(
            "Vista inicial",
            list(_v57_camera_presets().keys()),
            index=0,
            key="v57_initial_view",
        )
    with q2:
        trajectory = st.selectbox(
            "Trayectoria 3D",
            ["Ninguna"] + landmark_options,
            index=0,
            key="v57_trajectory_landmark",
            help="Muestra la trayectoria completa del landmark elegido durante el ciclo.",
        )
    with q3:
        show_labels = st.toggle(
            "Etiquetas de landmarks",
            value=False,
            key="v57_show_labels",
            help="Útil para auditoría. Puede recargar visualmente el modelo.",
        )

    st.markdown(
        "**Interacción:** arrastra sobre el modelo para rotarlo 360°, usa la rueda para zoom, "
        "arrastra con el botón secundario para desplazar la cámara y utiliza el slider inferior "
        "para ir a cualquier fase del ciclo."
    )

    try:
        fig = _v57_build_figure(
            accepted,
            show_labels=show_labels,
            trajectory_landmark=trajectory,
            initial_view=initial_view,
        )
        if fig is None:
            st.warning("No se pudo construir el visor dinámico con los puntos disponibles.")
            return

        st.plotly_chart(
            fig,
            use_container_width=True,
            config={
                "displaylogo": False,
                "scrollZoom": True,
                "responsive": True,
                "modeBarButtonsToRemove": ["toImage"],
            },
        )
    except Exception as exc:
        st.error(f"V57 no pudo construir el visor 3D interactivo: {exc}")
        with st.expander("Detalles técnicos V57"):
            st.code(traceback.format_exc())
        return

    nph = len(phases)
    npts = len(accepted)
    cycle_s = float(summary.get("physical_cycle_duration_s", summary.get("front_cycle_s", np.nan)))
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Instantes animados", str(nph))
    c2.metric("Puntos 3D utilizados", str(npts))
    c3.metric("Landmarks", str(int(accepted["landmark"].nunique())))
    c4.metric("Ciclo", f"{cycle_s:.2f} s" if np.isfinite(cycle_s) else "—")

    st.info(
        "La rotación, zoom y reproducción son únicamente herramientas de visualización. "
        "V57 no modifica las coordenadas 3D originales ni los criterios de aceptación V56."
    )


try:
    _v57_render_panel()
except Exception as _v57_exc:
    st.caption(f"V57 · visor 3D dinámico no disponible: {_v57_exc}")



# === V60 · 3D estimado/no métrico desde dos vistas sincronizadas ===
def _v60_estimated_3d_cycle():
    """Fusiona frontal (X,Y) y lateral (Z,Y) en coordenadas normalizadas.
    Es una representación espacial cualitativa: NO triangula rayos, NO usa una
    escala métrica y NO sustituye a V56 calibrada.
    """
    df1 = st.session_state.get("biplanar_seg_front")
    df2 = st.session_state.get("biplanar_seg_lateral")
    fps1 = float(st.session_state.get("biplanar_fps_front", 0.0) or 0.0)
    fps2 = float(st.session_state.get("biplanar_fps_lateral", 0.0) or 0.0)
    if not isinstance(df1, pd.DataFrame) or df1.empty:
        df1 = st.session_state.get("analysis_df")
        fps1 = float((st.session_state.get("meta1") or {}).get("fps", 0.0) or 0.0)
    if not isinstance(df2, pd.DataFrame) or df2.empty:
        df2 = st.session_state.get("analysis_df2")
        fps2 = float((st.session_state.get("meta2") or {}).get("fps", 0.0) or 0.0)
    v48 = st.session_state.get("biplanar_v48_profile") or {}
    v48s = (v48.get("summary") or {}) if isinstance(v48, dict) else {}
    if not (isinstance(df1,pd.DataFrame) and not df1.empty and isinstance(df2,pd.DataFrame) and not df2.empty and fps1>0 and fps2>0):
        return None, "Faltan los segmentos HALPE26 de las dos cámaras."
    if v48s.get("front_ic_frame") is None or v48s.get("front_next_ic_frame") is None:
        return None, "Todavía no existe un ciclo físico homólogo V48."
    sync = st.session_state.get("biplanar_v46_physical_clock_offset_s")
    try: sync=float(sync)
    except Exception: sync=np.nan
    if not np.isfinite(sync):
        return None, "Falta el reloj físico maestro entre cámaras."
    times=_v56_build_time_grid(v48s,df1,fps1,df2,fps2,sync)
    if len(times)<2:
        return None, "No existe intervalo físico común suficiente."
    lr_swapped=bool(v48s.get("lr_swapped",False))
    rows=[]
    # Escala independiente por vista: mediana hombros-cadera. Evita mezclar píxeles
    # de cámaras distintas; conserva sólo proporciones corporales aproximadas.
    def obs_at(t,name,cam):
        if cam==1: return _v56_interp_landmark(df1,fps1,t,name,0.0)
        return _v56_interp_landmark(df2,fps2,t,_v56_cam2_landmark_name(name,lr_swapped),sync)
    def center_scale(t,cam):
        hip=obs_at(t,"Hip",cam)
        neck=obs_at(t,"Neck",cam)
        if hip and neck:
            sc=float(np.hypot(neck["x"]-hip["x"],neck["y"]-hip["y"]))
            if np.isfinite(sc) and sc>5: return hip,sc
        lh=obs_at(t,"LHip",cam); rh=obs_at(t,"RHip",cam)
        ls=obs_at(t,"LShoulder",cam); rs=obs_at(t,"RShoulder",cam)
        if all([lh,rh,ls,rs]):
            hx=(lh["x"]+rh["x"])/2; hy=(lh["y"]+rh["y"])/2
            sx=(ls["x"]+rs["x"])/2; sy=(ls["y"]+rs["y"])/2
            sc=float(np.hypot(sx-hx,sy-hy))
            return {"x":hx,"y":hy}, sc if sc>5 else np.nan
        return None,np.nan
    for i,t in enumerate(times):
        c1,s1=center_scale(t,1); c2,s2=center_scale(t,2)
        if c1 is None or c2 is None or not np.isfinite([s1,s2]).all(): continue
        ph=100.0*i/max(1,len(times)-1)
        for name in HALPE26:
            a=obs_at(t,name,1); b=obs_at(t,name,2)
            if a is None or b is None: continue
            score=min(float(a["score"]),float(b["score"]))
            # X frontal, Z lateral; Y combina ambas alturas normalizadas.
            x=(float(a["x"])-float(c1["x"]))/s1
            yf=-(float(a["y"])-float(c1["y"]))/s1
            z=(float(b["x"])-float(c2["x"]))/s2
            yl=-(float(b["y"])-float(c2["y"]))/s2
            y=float(np.nanmean([yf,yl]))
            rows.append({"physical_time_s":float(t),"phase_pct":ph,"landmark":name,
                         "X":x,"Y":y,"Z":z,"accepted_3d":bool(score>=0.20),
                         "cam01_score":float(a["score"]),"cam02_score":float(b["score"]),
                         "reproj_mean_px":np.nan,"ray_angle_deg":np.nan,
                         "coordinate_type":"estimado_no_metrico"})
    pts=pd.DataFrame(rows)
    if pts.empty: return None,"No se pudieron fusionar landmarks coincidentes."
    return pts,None


def _v60_render_estimated_panel():
    level=str(st.session_state.get("analysis_level", ""))
    if level.startswith("Nivel 1"):
        return
    # En Nivel 3 calibrado se muestra como respaldo/comparación; nunca sustituye
    # silenciosamente a la triangulación calibrada.
    st.markdown("---")
    st.subheader("V60 · 3D estimado/no métrico")
    st.caption(
        "Reconstrucción visual aproximada obtenida al fusionar la coordenada frontal X–Y con la lateral Z–Y "
        "sobre el mismo reloj físico. No usa calibración, no triangula rayos y sus X/Y/Z son unidades normalizadas, no cm/mm."
    )
    pts,err=_v60_estimated_3d_cycle()
    if err:
        st.info(f"3D estimado todavía no disponible: {err}")
        return
    st.session_state["v60_estimated_3d"]={"points3d":pts,"method":"fusion_orthographic_normalized_front_lateral"}
    valid=pts[pts["accepted_3d"]==True].copy()
    if valid.empty:
        st.warning("No hay landmarks con confianza suficiente para el 3D estimado.")
        return
    q1,q2,q3=st.columns([1.1,1.2,1.4])
    with q1:
        estimated_cameras = _v62c_estimated_camera_presets()
        initial_view=st.selectbox("Vista inicial · 3D estimado",list(estimated_cameras.keys()),index=0,key="v60_initial_view")
    with q2:
        opts=sorted(valid["landmark"].dropna().astype(str).unique().tolist())
        trajectory=st.selectbox("Trayectoria · 3D estimado",["Ninguna"]+opts,index=0,key="v60_trajectory")
    with q3:
        labels=st.toggle("Etiquetas · 3D estimado",value=False,key="v60_labels")
    fig=_v57_build_figure(valid,show_labels=labels,trajectory_landmark=trajectory,initial_view=initial_view,camera_presets=estimated_cameras)
    if fig is not None:
        fig.update_layout(title=dict(text="V60 · Esqueleto 3D estimado · NO MÉTRICO",x=0.02))
        st.plotly_chart(fig,use_container_width=True,config={"displaylogo":False,"scrollZoom":True,"responsive":True,"modeBarButtonsToRemove":["toImage"]})
    c1,c2,c3=st.columns(3)
    c1.metric("Instantes",str(int(valid["phase_pct"].nunique())))
    c2.metric("Landmarks",str(int(valid["landmark"].nunique())))
    c3.metric("Unidad espacial","normalizada")
    st.warning(
        "3D ESTIMADO / NO MÉTRICO: sirve para visualización espacial cualitativa. No utilizar estas coordenadas "
        "para distancias, desplazamientos o ángulos articulares 3D cuantitativos. Para ello usa Nivel 3 con calibración válida."
    )

try:
    _v60_render_estimated_panel()
except Exception as _v60_exc:
    st.caption(f"V60 · 3D estimado no disponible: {_v60_exc}")


# === V58 CORREGIDA · Calibración clínica estéreo ChArUco ===

def _v58_cv2_aruco():
    import cv2
    if not hasattr(cv2, "aruco"):
        raise RuntimeError(
            "OpenCV no incluye el módulo aruco. Instala opencv-contrib-python-headless."
        )
    return cv2


def _v58_dictionary(cv2, name):
    mapping = {
        "DICT_4X4_50": cv2.aruco.DICT_4X4_50,
        "DICT_4X4_100": cv2.aruco.DICT_4X4_100,
        "DICT_5X5_100": cv2.aruco.DICT_5X5_100,
        "DICT_5X5_250": cv2.aruco.DICT_5X5_250,
        "DICT_6X6_250": cv2.aruco.DICT_6X6_250,
        "DICT_6X6_1000": cv2.aruco.DICT_6X6_1000,
    }
    if name not in mapping:
        raise ValueError(f"Diccionario ArUco no soportado: {name}")
    return cv2.aruco.getPredefinedDictionary(mapping[name])


def _v58_create_board(cv2, squares_x, squares_y, square_length_mm, marker_length_mm, dict_name):
    dictionary = _v58_dictionary(cv2, dict_name)
    sx = int(squares_x)
    sy = int(squares_y)
    sl = float(square_length_mm)
    ml = float(marker_length_mm)

    if sx < 3 or sy < 3:
        raise ValueError("El tablero debe tener al menos 3×3 cuadrados.")
    if not (sl > 0 and ml > 0 and ml < sl):
        raise ValueError("marker_length debe ser positivo y menor que square_length.")

    try:
        board = cv2.aruco.CharucoBoard((sx, sy), sl, ml, dictionary)
    except Exception:
        board = cv2.aruco.CharucoBoard_create(sx, sy, sl, ml, dictionary)
    return board, dictionary


def _v58_board_png(squares_x, squares_y, square_length_mm, marker_length_mm, dict_name):
    cv2 = _v58_cv2_aruco()
    board, _ = _v58_create_board(
        cv2, squares_x, squares_y, square_length_mm, marker_length_mm, dict_name
    )

    # Imagen de alta resolución manteniendo la proporción física del tablero.
    px_per_square = 220
    width = int(squares_x) * px_per_square
    height = int(squares_y) * px_per_square
    margin = max(20, int(px_per_square * 0.20))

    try:
        img = board.generateImage((width, height), marginSize=margin, borderBits=1)
    except Exception:
        img = cv2.aruco.drawPlanarBoard(board, (width, height), margin, 1)

    ok, enc = cv2.imencode(".png", img)
    if not ok:
        raise RuntimeError("No se pudo generar la imagen del tablero ChArUco.")
    return enc.tobytes()


def _v58_detect_charuco(gray, board, dictionary, cv2):
    """
    Devuelve esquinas ChArUco e IDs. Incluye compatibilidad con API nueva/antigua.
    """
    corners = None
    ids = None

    # API OpenCV moderna
    if hasattr(cv2.aruco, "CharucoDetector"):
        try:
            detector = cv2.aruco.CharucoDetector(board)
            cc, ci, _, _ = detector.detectBoard(gray)
            if cc is not None and ci is not None and len(ci) >= 4:
                return cc.astype("float32"), ci.astype("int32")
        except Exception:
            pass

    # API clásica
    marker_corners, marker_ids, _ = cv2.aruco.detectMarkers(gray, dictionary)
    if marker_ids is None or len(marker_ids) == 0:
        return None, None
    try:
        n, cc, ci = cv2.aruco.interpolateCornersCharuco(
            marker_corners, marker_ids, gray, board
        )
        if cc is not None and ci is not None and int(n) >= 4:
            return cc.astype("float32"), ci.astype("int32")
    except Exception:
        return None, None

    return None, None


def _v58_temp_video(uploaded, suffix=".mp4"):
    import tempfile
    fh = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    fh.write(uploaded.getvalue())
    fh.flush()
    fh.close()
    return fh.name


def _v58_video_meta(path, cv2):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(f"No se pudo abrir el vídeo: {path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    cap.release()
    if fps <= 0 or n <= 0 or w <= 0 or h <= 0:
        raise RuntimeError("Metadatos de vídeo incompletos.")
    return {"fps": fps, "frames": n, "width": w, "height": h, "duration_s": n / fps}


def _v58_read_frame_at(cap, frame_index, cv2):
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
    ok, frame = cap.read()
    return frame if ok else None


def _v58_calibrate_intrinsics(corners_list, ids_list, board, image_size, cv2):
    if len(corners_list) < 4:
        raise RuntimeError("Se necesitan al menos 4 capturas ChArUco válidas por cámara.")

    # API disponible en OpenCV contrib.
    if hasattr(cv2.aruco, "calibrateCameraCharucoExtended"):
        out = cv2.aruco.calibrateCameraCharucoExtended(
            charucoCorners=corners_list,
            charucoIds=ids_list,
            board=board,
            imageSize=image_size,
            cameraMatrix=None,
            distCoeffs=None,
        )
        rms, K, D = out[0], out[1], out[2]
    else:
        rms, K, D, _, _ = cv2.aruco.calibrateCameraCharuco(
            charucoCorners=corners_list,
            charucoIds=ids_list,
            board=board,
            imageSize=image_size,
            cameraMatrix=None,
            distCoeffs=None,
        )
    return float(rms), K, D


def _v58_board_object_points(board):
    try:
        pts = board.getChessboardCorners()
    except Exception:
        pts = board.chessboardCorners
    return np.asarray(pts, dtype=np.float32).reshape(-1, 3)


def _v58_common_charuco(cc1, ci1, cc2, ci2, board):
    ids1 = np.asarray(ci1).reshape(-1).astype(int)
    ids2 = np.asarray(ci2).reshape(-1).astype(int)
    p1 = np.asarray(cc1).reshape(-1, 2)
    p2 = np.asarray(cc2).reshape(-1, 2)

    common = sorted(set(ids1.tolist()).intersection(ids2.tolist()))
    if len(common) < 6:
        return None

    map1 = {int(i): p1[k] for k, i in enumerate(ids1)}
    map2 = {int(i): p2[k] for k, i in enumerate(ids2)}
    obj_all = _v58_board_object_points(board)

    valid_ids = [i for i in common if 0 <= i < len(obj_all)]
    if len(valid_ids) < 6:
        return None

    obj = np.asarray([obj_all[i] for i in valid_ids], dtype=np.float32)
    img1 = np.asarray([map1[i] for i in valid_ids], dtype=np.float32)
    img2 = np.asarray([map2[i] for i in valid_ids], dtype=np.float32)
    return obj, img1, img2, valid_ids


def _v58_epipolar_error(F, pts1, pts2, cv2):
    """
    Distancia epipolar simétrica media en píxeles sobre correspondencias observadas.
    """
    if F is None or len(pts1) == 0:
        return np.nan
    p1 = np.asarray(pts1, dtype=np.float32).reshape(-1, 1, 2)
    p2 = np.asarray(pts2, dtype=np.float32).reshape(-1, 1, 2)
    l2 = cv2.computeCorrespondEpilines(p1, 1, F).reshape(-1, 3)
    l1 = cv2.computeCorrespondEpilines(p2, 2, F).reshape(-1, 3)

    a1 = l1[:, 0]; b1 = l1[:, 1]; c1 = l1[:, 2]
    a2 = l2[:, 0]; b2 = l2[:, 1]; c2 = l2[:, 2]
    q1 = p1.reshape(-1, 2)
    q2 = p2.reshape(-1, 2)

    d1 = np.abs(a1*q1[:,0] + b1*q1[:,1] + c1) / np.maximum(np.hypot(a1,b1), 1e-12)
    d2 = np.abs(a2*q2[:,0] + b2*q2[:,1] + c2) / np.maximum(np.hypot(a2,b2), 1e-12)
    vals = np.concatenate([d1, d2])
    vals = vals[np.isfinite(vals)]
    return float(np.mean(vals)) if len(vals) else np.nan


def _v58_profile_to_json(profile):
    def conv(v):
        if isinstance(v, np.ndarray):
            return v.tolist()
        if isinstance(v, (np.floating, np.integer)):
            return v.item()
        if isinstance(v, dict):
            return {k: conv(x) for k, x in v.items()}
        if isinstance(v, (list, tuple)):
            return [conv(x) for x in v]
        return v
    return json.dumps(conv(profile), ensure_ascii=False, indent=2)


def _v58_profile_from_json(text):
    raw = json.loads(text)
    if not isinstance(raw, dict) or raw.get("profile_type") != "PhysioSentinel stereo calibration":
        raise ValueError("El JSON no corresponde a un perfil estéreo PhysioSentinel V58.")
    return raw


def _v58_run_calibration(
    file1, file2,
    squares_x, squares_y,
    square_length_mm, marker_length_mm,
    dict_name,
    sample_count=35,
    min_common_corners=6,
):
    import os
    cv2 = _v58_cv2_aruco()
    board, dictionary = _v58_create_board(
        cv2, squares_x, squares_y, square_length_mm, marker_length_mm, dict_name
    )

    p1 = _v58_temp_video(file1, suffix=Path(file1.name).suffix or ".mp4")
    p2 = _v58_temp_video(file2, suffix=Path(file2.name).suffix or ".mp4")

    try:
        m1 = _v58_video_meta(p1, cv2)
        m2 = _v58_video_meta(p2, cv2)

        if (m1["width"], m1["height"]) != (m2["width"], m2["height"]):
            raise RuntimeError(
                "V58 requiere la misma resolución en ambas cámaras para esta primera versión."
            )

        common_duration = min(m1["duration_s"], m2["duration_s"])
        if common_duration <= 0.5:
            raise RuntimeError("La secuencia de calibración común es demasiado corta.")

        n_samples = int(np.clip(sample_count, 8, 120))
        times = np.linspace(0.10, max(0.11, common_duration - 0.10), n_samples)

        cap1 = cv2.VideoCapture(p1)
        cap2 = cv2.VideoCapture(p2)

        c1_all, i1_all = [], []
        c2_all, i2_all = [], []
        stereo_obj, stereo_img1, stereo_img2 = [], [], []
        detection_rows = []

        for idx, t in enumerate(times):
            fidx1 = int(round(float(t) * m1["fps"]))
            fidx2 = int(round(float(t) * m2["fps"]))
            fr1 = _v58_read_frame_at(cap1, fidx1, cv2)
            fr2 = _v58_read_frame_at(cap2, fidx2, cv2)

            if fr1 is None or fr2 is None:
                detection_rows.append({
                    "sample": idx, "time_s": float(t),
                    "cam01_corners": 0, "cam02_corners": 0,
                    "common_corners": 0, "stereo_pair_used": False
                })
                continue

            g1 = cv2.cvtColor(fr1, cv2.COLOR_BGR2GRAY)
            g2 = cv2.cvtColor(fr2, cv2.COLOR_BGR2GRAY)
            cc1, ci1 = _v58_detect_charuco(g1, board, dictionary, cv2)
            cc2, ci2 = _v58_detect_charuco(g2, board, dictionary, cv2)

            n1 = 0 if ci1 is None else int(len(ci1))
            n2 = 0 if ci2 is None else int(len(ci2))

            if cc1 is not None and ci1 is not None and n1 >= 4:
                c1_all.append(cc1); i1_all.append(ci1)
            if cc2 is not None and ci2 is not None and n2 >= 4:
                c2_all.append(cc2); i2_all.append(ci2)

            common_n = 0
            used = False
            if cc1 is not None and ci1 is not None and cc2 is not None and ci2 is not None:
                pair = _v58_common_charuco(cc1, ci1, cc2, ci2, board)
                if pair is not None:
                    obj, im1, im2, common_ids = pair
                    common_n = len(common_ids)
                    if common_n >= int(min_common_corners):
                        stereo_obj.append(obj)
                        stereo_img1.append(im1)
                        stereo_img2.append(im2)
                        used = True

            detection_rows.append({
                "sample": idx,
                "time_s": float(t),
                "cam01_corners": n1,
                "cam02_corners": n2,
                "common_corners": int(common_n),
                "stereo_pair_used": bool(used),
            })

        cap1.release()
        cap2.release()

        image_size = (int(m1["width"]), int(m1["height"]))

        if len(c1_all) < 6 or len(c2_all) < 6:
            raise RuntimeError(
                f"Detecciones insuficientes: cam01={len(c1_all)}, cam02={len(c2_all)}. "
                "Mueve el tablero por distintas posiciones, distancias e inclinaciones."
            )
        if len(stereo_obj) < 6:
            raise RuntimeError(
                f"Solo hay {len(stereo_obj)} pares estéreo útiles. "
                "Ambas cámaras deben ver el tablero simultáneamente en varias poses."
            )

        rms1, K1, D1 = _v58_calibrate_intrinsics(c1_all, i1_all, board, image_size, cv2)
        rms2, K2, D2 = _v58_calibrate_intrinsics(c2_all, i2_all, board, image_size, cv2)

        flags = cv2.CALIB_FIX_INTRINSIC
        stereo_out = cv2.stereoCalibrate(
            stereo_obj,
            stereo_img1,
            stereo_img2,
            K1, D1,
            K2, D2,
            image_size,
            flags=flags,
            criteria=(
                cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
                100,
                1e-7,
            ),
        )

        stereo_rms, K1s, D1s, K2s, D2s, R, T, E, F = stereo_out[:9]

        baseline_mm = float(np.linalg.norm(np.asarray(T, dtype=float).reshape(3)))
        C1 = np.zeros(3, dtype=float)
        C2 = (-np.asarray(R, dtype=float).T @ np.asarray(T, dtype=float).reshape(3,1)).reshape(3)

        # Error epipolar sobre todos los pares realmente usados.
        epi_vals = []
        for im1, im2 in zip(stereo_img1, stereo_img2):
            e = _v58_epipolar_error(F, im1, im2, cv2)
            if np.isfinite(e):
                epi_vals.append(e)
        epi_mean = float(np.mean(epi_vals)) if epi_vals else np.nan
        epi_p95 = float(np.percentile(epi_vals, 95)) if epi_vals else np.nan

        detection_df = pd.DataFrame(detection_rows)

        profile = {
            "profile_type": "PhysioSentinel stereo calibration",
            "version": "58",
            "model": "OpenCV pinhole + Brown-Conrady/radtan",
            "board": {
                "type": "ChArUco",
                "dictionary": dict_name,
                "squares_x": int(squares_x),
                "squares_y": int(squares_y),
                "square_length_mm": float(square_length_mm),
                "marker_length_mm": float(marker_length_mm),
            },
            "capture": {
                "image_width_px": int(image_size[0]),
                "image_height_px": int(image_size[1]),
                "cam01_fps": float(m1["fps"]),
                "cam02_fps": float(m2["fps"]),
                "cam01_source": file1.name,
                "cam02_source": file2.name,
                "sample_count_requested": int(n_samples),
                "cam01_valid_views": int(len(c1_all)),
                "cam02_valid_views": int(len(c2_all)),
                "stereo_valid_pairs": int(len(stereo_obj)),
            },
            "quality": {
                "intrinsic_rms_cam01_px": float(rms1),
                "intrinsic_rms_cam02_px": float(rms2),
                "stereo_rms_px": float(stereo_rms),
                "epipolar_mean_px": epi_mean,
                "epipolar_p95_px": epi_p95,
                "baseline_mm": baseline_mm,
                "quality_note": (
                    "Métricas geométricas internas de calibración. "
                    "V58 no las interpreta como umbrales clínicos validados."
                ),
            },
            "cam01": {
                "K": np.asarray(K1s, dtype=float),
                "dist_coeffs": np.asarray(D1s, dtype=float).reshape(-1),
                "R_world_to_camera": np.eye(3, dtype=float),
                "t_world_to_camera_mm": np.zeros(3, dtype=float),
                "camera_center_world_mm": C1,
            },
            "cam02": {
                "K": np.asarray(K2s, dtype=float),
                "dist_coeffs": np.asarray(D2s, dtype=float).reshape(-1),
                "R_world_to_camera": np.asarray(R, dtype=float),
                "t_world_to_camera_mm": np.asarray(T, dtype=float).reshape(3),
                "camera_center_world_mm": C2,
            },
            "stereo": {
                "R_cam01_to_cam02": np.asarray(R, dtype=float),
                "T_cam01_to_cam02_mm": np.asarray(T, dtype=float).reshape(3),
                "E": np.asarray(E, dtype=float),
                "F": np.asarray(F, dtype=float),
                "baseline_mm": baseline_mm,
            },
            "reuse_conditions": [
                "Mismas cámaras.",
                "Misma resolución y modo de captura.",
                "Mismo zoom/focal efectiva.",
                "Misma posición y orientación relativa entre cámaras.",
            ],
            "integration_status": (
                "Perfil listo para integración con el triangulador clínico OpenCV. "
                "V56 Tsai/GPJATK permanece congelada y no se sustituye silenciosamente."
            ),
        }

        return {
            "profile": profile,
            "detection_df": detection_df,
            "summary": {
                "version": "58",
                "calibration_valid": True,
                "cam01_valid_views": int(len(c1_all)),
                "cam02_valid_views": int(len(c2_all)),
                "stereo_valid_pairs": int(len(stereo_obj)),
                "intrinsic_rms_cam01_px": float(rms1),
                "intrinsic_rms_cam02_px": float(rms2),
                "stereo_rms_px": float(stereo_rms),
                "epipolar_mean_px": epi_mean,
                "epipolar_p95_px": epi_p95,
                "baseline_mm": baseline_mm,
                "image_size_px": [int(image_size[0]), int(image_size[1])],
                "camera_model": "OpenCV pinhole + Brown-Conrady/radtan",
            },
        }

    finally:
        for p in [p1, p2]:
            try:
                os.unlink(p)
            except Exception:
                pass


def _v58_matrix_df(M):
    a = np.asarray(M, dtype=float)
    return pd.DataFrame(a, columns=[f"c{i+1}" for i in range(a.shape[1])])


def _v58_render_profile(profile):
    if not isinstance(profile, dict):
        return
    q = profile.get("quality") or {}
    cap = profile.get("capture") or {}

    st.success("Perfil de calibración estéreo disponible.")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("RMS intrínseco cam01", f"{q.get('intrinsic_rms_cam01_px', np.nan):.3f} px")
    c2.metric("RMS intrínseco cam02", f"{q.get('intrinsic_rms_cam02_px', np.nan):.3f} px")
    c3.metric("RMS estéreo", f"{q.get('stereo_rms_px', np.nan):.3f} px")
    c4.metric("Baseline", f"{q.get('baseline_mm', np.nan):.1f} mm")

    d1, d2, d3 = st.columns(3)
    d1.metric("Vistas válidas cam01", str(cap.get("cam01_valid_views", "—")))
    d2.metric("Vistas válidas cam02", str(cap.get("cam02_valid_views", "—")))
    d3.metric("Pares estéreo válidos", str(cap.get("stereo_valid_pairs", "—")))

    st.caption(
        f"Error epipolar medio: {q.get('epipolar_mean_px', np.nan):.3f} px · "
        f"P95: {q.get('epipolar_p95_px', np.nan):.3f} px."
    )

    with st.expander("Matrices de calibración V58"):
        st.markdown("**K cam01**")
        st.dataframe(_v58_matrix_df(profile["cam01"]["K"]), use_container_width=True, hide_index=True)
        st.markdown("**K cam02**")
        st.dataframe(_v58_matrix_df(profile["cam02"]["K"]), use_container_width=True, hide_index=True)
        st.markdown("**R cam01→cam02**")
        st.dataframe(_v58_matrix_df(profile["stereo"]["R_cam01_to_cam02"]), use_container_width=True, hide_index=True)
        st.markdown("**T cam01→cam02 (mm)**")
        st.dataframe(
            pd.DataFrame({"T_mm": np.asarray(profile["stereo"]["T_cam01_to_cam02_mm"], dtype=float).reshape(-1)}),
            use_container_width=True,
            hide_index=True,
        )

    st.info(
        "Este perfil puede reutilizarse mientras no cambien las cámaras, resolución/zoom "
        "ni su posición relativa. Si mueves un trípode o modificas el zoom, recalibra."
    )


def _v58_render_panel():
    st.markdown("---")
    st.subheader("V58 · Calibración clínica estéreo")
    st.caption(
        "Crea una calibración propia para dos móviles/cámaras mediante un tablero ChArUco. "
        "Así los futuros estudios clínicos no dependen de los XML de GPJATK."
    )

    st.warning(
        "La calibración estéreo requiere que ambas cámaras vean el MISMO tablero en el MISMO instante. "
        "Graba los dos vídeos de calibración simultáneamente, sin mover las cámaras después."
    )

    with st.expander("1 · Crear / descargar tablero ChArUco", expanded=False):
        b1, b2, b3 = st.columns(3)
        with b1:
            squares_x = st.number_input("Cuadrados X", min_value=4, max_value=12, value=7, step=1, key="v58_sx")
            squares_y = st.number_input("Cuadrados Y", min_value=4, max_value=12, value=5, step=1, key="v58_sy")
        with b2:
            square_mm = st.number_input("Lado del cuadrado (mm)", min_value=10.0, max_value=100.0, value=40.0, step=1.0, key="v58_square_mm")
            marker_mm = st.number_input("Lado del marcador (mm)", min_value=5.0, max_value=90.0, value=30.0, step=1.0, key="v58_marker_mm")
        with b3:
            dict_name = st.selectbox(
                "Diccionario ArUco",
                ["DICT_5X5_100", "DICT_4X4_50", "DICT_4X4_100", "DICT_5X5_250", "DICT_6X6_250"],
                index=0,
                key="v58_dict",
            )

        st.caption(
            "Imprime al 100 % sin 'ajustar a página'. Después mide físicamente un cuadrado y, "
            "si el tamaño real difiere, introduce arriba el valor real antes de calibrar."
        )

        try:
            board_png = _v58_board_png(
                int(squares_x), int(squares_y),
                float(square_mm), float(marker_mm),
                dict_name,
            )
            st.image(board_png, caption="Vista previa del tablero ChArUco", use_container_width=False)
            st.download_button(
                "⬇️ Descargar tablero ChArUco PNG",
                data=board_png,
                file_name=f"PhysioSentinel_ChArUco_{int(squares_x)}x{int(squares_y)}_{int(square_mm)}mm.png",
                mime="image/png",
                use_container_width=True,
            )
        except Exception as exc:
            st.error(f"No se pudo generar el tablero: {exc}")

    st.markdown("### 2 · Vídeos de calibración simultáneos")
    a1, a2 = st.columns(2)
    with a1:
        cal1 = st.file_uploader(
            "Vídeo calibración cam01 (frontal)",
            type=["mp4", "mov", "avi", "mkv"],
            key="v58_cal_video1",
        )
    with a2:
        cal2 = st.file_uploader(
            "Vídeo calibración cam02 (lateral)",
            type=["mp4", "mov", "avi", "mkv"],
            key="v58_cal_video2",
        )

    c1, c2 = st.columns(2)
    with c1:
        sample_count = st.slider(
            "Muestras temporales a analizar",
            min_value=12, max_value=80, value=35, step=1,
            key="v58_sample_count",
            help="Se distribuyen uniformemente durante el tramo común de ambos vídeos.",
        )
    with c2:
        st.markdown(
            "**Durante la grabación:** mueve el tablero por el volumen donde caminará el paciente, "
            "incluyendo centro, laterales, distintas alturas, distancias e inclinaciones."
        )

    can_run = cal1 is not None and cal2 is not None
    if st.button(
        "🎯 Calibrar cámaras",
        type="primary",
        use_container_width=True,
        disabled=not can_run,
        key="v58_run_calibration",
    ):
        try:
            with st.spinner("Detectando ChArUco, calibrando intrínsecos y resolviendo geometría estéreo..."):
                out = _v58_run_calibration(
                    cal1, cal2,
                    int(squares_x), int(squares_y),
                    float(square_mm), float(marker_mm),
                    dict_name,
                    sample_count=int(sample_count),
                )
            st.session_state["v58_calibration"] = out
            st.session_state["v58_active_profile"] = out["profile"]
            st.success("Calibración V58 completada.")
        except Exception as exc:
            st.error(f"V58 no pudo completar la calibración: {exc}")
            with st.expander("Detalles técnicos"):
                st.code(traceback.format_exc())

    out = st.session_state.get("v58_calibration")
    if isinstance(out, dict):
        profile = out.get("profile")
        _v58_render_profile(profile)

        det = out.get("detection_df")
        if isinstance(det, pd.DataFrame) and not det.empty:
            with st.expander("Control de detecciones ChArUco"):
                st.dataframe(det, use_container_width=True, hide_index=True)

        if isinstance(profile, dict):
            profile_json = _v58_profile_to_json(profile)
            st.download_button(
                "💾 Guardar perfil de calibración V58",
                data=profile_json.encode("utf-8"),
                file_name="PhysioSentinel_perfil_calibracion_estereo_V58.json",
                mime="application/json",
                use_container_width=True,
            )

            if isinstance(det, pd.DataFrame):
                st.download_button(
                    "⬇️ Descargar QC de calibración CSV",
                    data=det.to_csv(index=False).encode("utf-8-sig"),
                    file_name="PhysioSentinel_QC_calibracion_V58.csv",
                    mime="text/csv",
                    use_container_width=True,
                )

    st.markdown("### 3 · Reutilizar una calibración guardada")
    saved = st.file_uploader(
        "Cargar perfil V58 (.json)",
        type=["json"],
        key="v58_saved_profile",
        help="Permite reutilizar una calibración sin volver a grabar el tablero.",
    )
    if saved is not None:
        try:
            loaded = _v58_profile_from_json(saved.getvalue().decode("utf-8"))
            if st.button("Activar perfil cargado", use_container_width=True, key="v58_activate_saved"):
                st.session_state["v58_active_profile"] = loaded
                st.success("Perfil V58 cargado y activado.")
            with st.expander("Resumen del perfil cargado"):
                _v58_render_profile(loaded)
        except Exception as exc:
            st.error(f"Perfil V58 no válido: {exc}")

    active = st.session_state.get("v58_active_profile")
    if isinstance(active, dict):
        st.success(
            "Perfil clínico V58 activo en memoria. Ya no dependes de XML externos para conservar "
            "la geometría de este montaje."
        )

    st.info(
        "V58 resuelve y guarda la calibración clínica con modelo OpenCV pinhole + Brown-Conrady. "
        "La ruta Tsai V55/V56 de GPJATK permanece congelada para no alterar la validación ya realizada. "
        "La integración del perfil V58 con la triangulación clínica se hará de forma explícita, "
        "sin convertirlo silenciosamente a Tsai."
    )


try:
    _v58_render_panel()
except Exception as _v58_exc:
    st.caption(f"V58 · calibración clínica estéreo no disponible: {_v58_exc}")
