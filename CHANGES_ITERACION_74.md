# PhysioSentinel Gait · Iteración 74

## Cambio principal
La máscara automática de marcha rectilínea pasa a ser una **máscara maestra global**. Los frames correspondientes a cambio de sentido, transición y bordes excluidos ya no participan en la cadena biomecánica posterior.

## Implementación
- La detección automática de giro se conserva sin modificar sus fórmulas.
- Los ciclos IC→IC deben quedar completamente dentro del dominio rectilíneo válido (100 %) para ser aceptados.
- El detector cinemático de alternancia se ejecuta de forma independiente en cada bloque rectilíneo continuo; no interpola a través del giro.
- Tras finalizar la cadena temporal, coordenadas, scores y variables derivadas de frames no rectilíneos se convierten a NaN antes de las métricas espaciales.
- La máscara gobierna pelvis, hombros, tronco, rodilla, pie/retropié, COM, COM/BOS, velocidad lateral, valgo proyectado, perfiles por fase, consistencia cíclica y acoplamiento tronco-pelvis.
- El acoplamiento tronco-pelvis se calcula por bloques rectilíneos para evitar interpolación a través de un giro; se conserva la fórmula existente dentro de cada bloque.
- Las gráficas mantienen huecos reales en los tramos excluidos y los PNG exportados sombrean la zona como “Cambio de sentido / transición excluida”.
- `datos_graficos.csv` incorpora `straight_walking_valid` y `straight_block_id` para auditoría.
- El QC de tracking/visibilidad publicado se calcula sobre el mismo dominio rectilíneo utilizable.
- Se añaden las métricas de control `straight_walking_blocks` y `straight_mask_global_flag`.
- Los textos de informe y notas metodológicas heredan las métricas ya filtradas; no se modifican las fórmulas biomecánicas existentes.

## No modificado
No se cambian las definiciones geométricas/biomecánicas, el modelo HALPE26/RTMPose, calibración, triangulación 3D ni las fórmulas originales de las variables. La modificación afecta al **dominio temporal sobre el que dichas fórmulas reciben datos**.
