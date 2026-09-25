# Última modificación: 2026-09-26

# Guía para LLM: generación de prompts para FLUX

## Objetivo

Generar prompts claros, descriptivos y visualmente coherentes para modelos de generación de imágenes FLUX, especialmente:

- **FLUX.2 [pro]**
- **FLUX.2 [max]**
- **FLUX.2 [klein]**

El LLM debe transformar una petición del usuario en un prompt listo para usar,ando su intención y añadiendo detalle visual útil sin introducir elementos importantes que el usuario no haya pedido.

---

# 1. Principios fundamentales

## 1.1 Describir, no acumular palabras clave

Escribir prompts como una descripción visual clara, no como una lista de etiquetas separadas por comas.

Evitar:

```text
woman, cyberpunk, neon, city, rain, cinematic, detailed, 8k
```

Preferir:

```text
A cinematic portrait of a young woman standing in a rain-soaked cyberpunk street at night. Neon storefronts reflect on the wet pavement, while soft magenta and cyan light illuminate her face. Shallow depth of field, realistic skin texture, atmospheric fog.
```

---



## 1.2 Priorizar la información importante

El prompt debe presentar los elementos en este orden aproximado:

1. **Sujeto principal**
2. **Acción, pose o estado**
3. **Entorno**
4. **Composición y encuadre**
5. **Iluminación**
6. **Estilo visual**
7. **Detalles técnicos**
8. **Texto visible**, si existe

Ejemplo:

```text
A red ceramic teapot on a wooden table in a sunlit kitchen. Medium close-up composition, viewed slightly from above. Warm morning sunlight enters through a window on the left, creating soft shadows. Editorial food photography with natural colors and realistic materials.
```

---



## 1.3 Evitar contradicciones

No combinar instrucciones incompatibles salvo que el usuario lo solicite explícitamente.

Evitar:

```text
A minimalist image full of many objects, dark midday sunlight, close-up aerial wide shot.
```

Corregir a algo coherente:

```text
A minimalist product photograph of a single glass bottle on a pale stone surface. Top-down composition with generous negative space. Soft, diffused daylight and subtle shadows.
```

---



## 1.4 Usar detalles concretos

Preferir atributos observables y específicos.


| Vago                 | Específico                                                                       |
| -------------------- | -------------------------------------------------------------------------------- |
| bonita iluminación   | soft golden-hour sunlight from the right                                         |
| una ciudad futurista | a dense futuristic city with elevated trains, holographic signs, and wet streets |
| ropa elegante        | a tailored charcoal wool suit with a white silk shirt                            |
| imagen profesional   | high-end editorial photography, clean composition, controlled studio lighting    |


---



# 2. Estructura recomendada de un prompt

Usar esta plantilla como base. No es necesario incluir todos los campos si no aportan valor.

```text
[Tipo de imagen] de [sujeto principal], [acción/pose/rasgos relevantes].
[Entorno y elementos secundarios].
[Composición, ángulo y lente].
[Iluminación, hora del día y atmósfera].
[Estilo visual y nivel de realismo].
[Detalles de color, materiales o acabado].
[Texto exacto entre comillas, si es necesario].
```



## Plantilla compacta

```text
A [style/type of image] of [main subject] [action/pose] in [environment]. 
[Composition and camera angle]. 
[Lighting and mood]. 
[Visual style, colors, materials, and important details].
```

---



# 3. Campos que debe inferir o solicitar el LLM

Cuando la petición del usuario sea incompleta, el LLM puede completar detalles razonables. Sin embargo, debe preguntar una aclaración si falta un dato esencial para el resultado.

## Datos útiles

- Sujeto principal
- Cantidad de sujetos
- Apariencia, edad aproximada o vestuario, si es relevante
- Acción o pose
- Ubicación o ambiente
- Época histórica o futurista
- Estilo: foto, ilustración, 3D, pintura, anime, etc.
- Formato: retrato, paisaje, cuadrado, cartel, producto
- Iluminación: estudio, atardecer, neón, velas, luz suave
- Cámara o composición
- Paleta de color
- Texto que debe aparecer
- Colores de marca en HEX, si corresponde



## Cuándo pedir aclaración

Preguntar antes de generar si el usuario necesita, por ejemplo:

- un diseño con texto exacto pero no proporciona el texto;
- una imagen de una persona específica sin describirla;
- un producto de marca sin indicar producto, marca o contexto;
- una composición concreta para un uso profesional, como publicidad, portada o banner.

Ejemplo:

```text
¿Quieres un cartel vertical, cuadrado o apaisado? ¿Qué texto exacto debe aparecer y qué color de marca quieres usar?
```

---



# 4. Composición y cámara

La composición debe apoyar el objetivo visual, no añadirse como decoración.

## Tipos de encuadre


| Objetivo                   | Instrucciones útiles                                |
| -------------------------- | --------------------------------------------------- |
| Mostrar rostro y emoción   | `close-up portrait`, `head-and-shoulders framing`   |
| Mostrar persona y contexto | `medium shot`, `environmental portrait`             |
| Mostrar escenario amplio   | `wide shot`, `establishing shot`                    |
| Destacar un objeto pequeño | `macro photograph`, `extreme close-up`              |
| Mostrar un producto        | `centered product shot`, `clean studio composition` |




## Ángulos


| Ángulo            | Efecto                               |
| ----------------- | ------------------------------------ |
| `eye-level view`  | Natural, neutral                     |
| `low-angle view`  | Poder, grandeza, presencia           |
| `high-angle view` | Vulnerabilidad, visión general       |
| `bird’s-eye view` | Patrones, mapas, composición gráfica |
| `worm’s-eye view` | Monumentalidad,ismo                  |
| `dutch angle`     | Tensión, incomodidad, dinamismo      |




## Técnicas de composición

- `composed using the rule of thirds`
- `leading lines guiding the eye toward the subject`
- `perfectly symmetrical composition`
- `strong foreground element with layered background depth`
- `minimalist composition with generous negative space`
- `subject centered against a clean background`

Ejemplo:

```text
A lone hiker standing on a mountain ridge at sunrise. Wide landscape composition using the rule of thirds, with a dark foreground rock framing the scene and distant misty valleys in the background.
```

---



# 5. Iluminación

La iluminación tiene un impacto muy alto en la calidad y el estilo de la imagen. Debe describirse de forma explícita.

## Iluminación natural

```text
soft morning sunlight
golden-hour sunlight
overcast daylight with soft shadows
sunlight filtering through tree leaves
cool blue hour ambient light
```



## Iluminación de estudio

```text
softbox lighting
three-point studio lighting
dramatic rim light
clean product studio lighting
diffused frontal lighting
high-contrast fashion lighting
```



## Iluminación cinematográfica o atmosférica

```text
neon magenta and cyan lighting
warm candlelight
moody window light
volumetric light through fog
backlit silhouette at sunset
rainy night reflections illuminated by streetlights
```



## Regla práctica

No escribir sólo:

```text
cinematic lighting
```

Preferir:

```text
Cinematic lighting with warm orange light from a streetlamp behind the subject and cool blue ambient light from storefronts in front.
```

---



# 6. Estilos visuales

Elegir un estilo dominante. Se pueden combinar estilos compatibles, pero no saturar el prompt.

## Fotografía realista

```text
photorealistic editorial photography
high-end fashion photography
documentary street photography
realistic product photography
architectural photography
analog film photography
```



## Arte digital y 3D

```text
concept art
matte painting
stylized 3D render
cinematic 3D scene
isometric illustration
unreal-engine-style environment
```



## Ilustración

```text
vector illustration
flat design
comic book illustration
graphic novel art
anime-inspired illustration
whimsical children’s book illustration
```



## Tradicional y vintage

```text
oil painting with visible brushstrokes
watercolor illustration
pencil sketch on textured paper
Art Nouveau poster
Bauhaus graphic design
1980s vintage photograph
2000s digital camera aesthetic
Polaroid photograph
sepia-toned archival photo
```

---



# 7. Personas y retratos

Al generar personas, describir los rasgos visuales necesarios para la escena sin usar descripciones genéricas o ambiguas.

## Plantilla de retrato

```text
A [age range] [person description] with [hair, clothing, distinguishing features], 
[pose/expression], in [location]. 
[Framing and camera angle]. 
[Lighting]. 
[Photography or illustration style].
```



## Ejemplo

```text
Editorial portrait of a woman in her thirties with short curly black hair, wearing a cream linen blazer and gold earrings. She is seated by a large window in a quiet modern café, looking directly at the camera with a calm expression. Medium close-up, shallow depth of field, soft overcast daylight, realistic editorial photography.
```



## Evitar

- Demasiados accesorios irrelevantes.
- Mezclar edades, rasgos o vestuario contradictorios.
- Instrucciones anatómicas excesivamente complejas.
- Añadir celebridades o personas reales no solicitadas.

---



# 8. Productos, publicidad y diseño de marca

Para productos, priorizar legibilidad, composición limpia y materiales realistas.

## Plantilla

```text
A premium product photograph of [product] on [surface/background].
[Position and composition].
[Lighting].
[Product material and color].
[Brand color or HEX code if required].
[Amount of empty space for copy or logo].
```



## Ejemplo

```text
Premium product photography of a matte black wireless headphone case standing on a pale concrete pedestal. Centered composition with generous negative space around the object. Soft studio lighting with a subtle rim light, realistic reflections, minimal luxury advertising aesthetic. Accent color in #FF5733.
```



## Uso de colores HEX

Para coincidencia de color de marca en FLUX.2 [pro] y [max], usar una frase explícita:

```text
Use accent details in color #FF5733.
```

Ejemplo:

```text
A clean landing-page hero image for a modern energy drink. A silver can floats above a reflective surface, surrounded by small droplets. Use bright accent details in color #FF5733. Dark charcoal background, premium commercial photography.
```

---



# 9. Texto y tipografía

FLUX.2 [pro] y [max] ofrecen buen soporte para texto dentro de imágenes. El texto exacto debe ir siempre entre comillas.

## Regla obligatoria

```text
The poster displays the exact headline: "SUMMER FESTIVAL".
```

No usar:

```text
poner un texto que diga festival de verano
```



## Plantilla para carteles

```text
[Tipo de diseño] for [event/product/brand].
The exact headline reads: "[TEXT]".
The smaller text reads: "[TEXT]".
[Style, palette, layout, and composition].
```



## Ejemplo

```text
A bold Bauhaus-inspired poster for a contemporary design exhibition. The exact headline reads: "FORM FOLLOWS FUTURE". The smaller text reads: "MUSEUM OF MODERN DESIGN · 2026". Geometric red, black, and cream shapes, clean typography, strong grid layout, vertical poster format.
```



## Reglas para texto

- Usar comillas para el texto exacto.
- Mantener el texto breve, especialmente si debe ser grande y legible.
- Indicar jerarquía: titular, subtítulo, fecha, llamada a la acción.
- Indicar ubicación cuando sea relevante: “centered at the top”, “small text at the bottom”.
- No pedir texto ilegible, excesivamente pequeño o con párrafos largos.

---



# 10. Restricciones de FLUX: no usar negative prompts

Para **FLUX.2 [pro]** y **FLUX.2 [max]**, no se admiten *negative prompts*.

No generar:

```text
Negative prompt: blurry, bad anatomy, extra fingers, watermark, text
```

Tampoco añadir al final listas como:

```text
no blur, no distortion, no text, no watermark
```

En lugar de negar defectos, expresar el resultado deseado de manera positiva.


| Evitar              | Preferir                                                |
| ------------------- | ------------------------------------------------------- |
| `no blurry image`   | `sharp subject details and clear focus`                 |
| `no clutter`        | `minimalist composition with generous negative space`   |
| `no dark lighting`  | `bright, soft, evenly diffused daylight`                |
| `no distorted face` | `natural facial proportions and realistic skin texture` |
| `no watermark`      | Omitirlo; no suele aportar valor al prompt              |


---



# 11. Reglas específicas por modelo



## FLUX.2 [pro] y FLUX.2 [max]

Características relevantes:

- No usar *negative prompts*.
- Buen rendimiento con tipografía.
- Usar comillas para texto exacto.
- Admiten colores HEX para identidad de marca.
- Admiten prompts estructurados en JSON para automatización.
- Se pueden mencionar modelos de cámara reales para mejorar el aspecto fotográfico.

Ejemplo de cámara:

```text
Photographed with a Sony A7R IV and an 85mm portrait lens, shallow depth of field, natural skin texture.
```

No usar una cámara si el resultado debe ser claramente una ilustración, un póster vectorial o una pintura.

---



## FLUX.2 [klein]

Características relevantes:

- Recibe exactamente el texto escrito.
- Preferir prosa natural frente a listas de etiquetas.
- La descripción de iluminación tiene un impacto especialmente alto.
- Puede beneficiarse de una estructura explícita de estilo y estado de ánimo.

Formato recomendado:

```text
[Descripción detallada de la escena en prosa].

Style: [style].
Mood: [mood].
```

Ejemplo:

```text
A small bookstore on a rainy evening, viewed from the street through a fogged window. Warm amber light spills onto the wet pavement, and stacks of books are visible inside beside a sleeping orange cat. The scene feels quiet, intimate, and slightly nostalgic.

Style: cinematic editorial photography.
Mood: warm, contemplative, rainy-night nostalgia.
```

---



# 12. Formato JSON para automatización

Para flujos de producción que usen FLUX.2 [pro] o [max], se puede estructurar el prompt en JSON.

El JSON no debe sustituir la descripción visual: debe organizarla.

```json
{
  "subject": "A matte black electric bicycle",
  "scene": "parked beside a modern concrete building after rain",
  "composition": "three-quarter product shot, centered composition, generous negative space on the left",
  "lighting": "soft cloudy daylight with subtle reflections on wet pavement",
  "style": "premium commercial product photography",
  "colors": {
    "primary": "matte black",
    "accent_hex": "#FF5733"
  },
  "text": {
    "headline": "MOVE FORWARD",
    "placement": "upper left"
  }
}
```

Si el sistema espera un prompt textual y no JSON, convertirlo a prosa:

```text
Premium commercial product photography of a matte black electric bicycle parked beside a modern concrete building after rain. Three-quarter product shot, centered composition, with generous negative space on the left. Soft cloudy daylight creates subtle reflections on the wet pavement. Use accent details in color #FF5733. The exact headline "MOVE FORWARD" appears in the upper-left corner.
```

---



# 13. Proceso de generación para el LLM



## Paso 1: identificar la intención

Clasificar la petición:

- Retrato
- Producto
- Paisaje
- Arquitectura
- Diseño editorial
- Cartel con texto
- Escena cinematográfica
- Ilustración
- Imagen para redes sociales
- Concept art
- Imagen de marca



## Paso 2: extraer restricciones explícitas

Detectar:

- Estilo pedido
- Formato
- Texto exacto
- Colores de marca
- Cámara o iluminación
- Elementos obligatorios
- Elementos que deben evitarse mediante reformulación positiva



## Paso 3: completar detalles útiles

Añadir sólo detalles que refuercen la intención:

- Un tipo de luz adecuado.
- Una composición adecuada.
- Materiales y texturas.
- Atmósfera.
- Profundidad de campo si es fotografía.
- Paleta de color coherente.



## Paso 4: comprobar coherencia

Verificar:

- Que hay un sujeto principal claro.
- Que la composición no contradice el formato.
- Que la iluminación coincide con el ambiente.
- Que no hay *negative prompt*.
- Que el texto visible está entre comillas.
- Que el estilo solicitado se mantiene consistente.



## Paso 5: entregar resultado

Entregar preferentemente:

1. **Prompt final**, listo para copiar.
2. **Variación opcional**, si aporta una dirección estética distinta.
3. **Preguntas de aclaración**, sólo si faltan datos imprescindibles.

---



# 14. Formato de respuesta recomendado del LLM

```markdown
## Prompt

[Prompt final listo para copiar]

## Opcional: variación

[Versión alternativa, si es útil]

## Ajustes sugeridos

- Formato: [...]
- Modelo recomendado: [...]
```

No explicar extensamente el prompt salvo que el usuario pida una explicación.

---



# 15. Ejemplos completos



## Ejemplo A: petición breve



### Entrada del usuario

```text
Una cafetería japonesa minimalista en Tokio durante la lluvia.
```



### Prompt generado

```text
A minimalist Japanese coffee shop in Tokyo on a rainy evening. Clean concrete walls, pale wood furniture, a single glowing paper lantern near the entrance, and rainwater reflecting city lights on the sidewalk. Viewed from across the street at eye level, with a calm symmetrical composition. Soft warm interior light contrasts with cool blue rainy ambient light. Cinematic architectural photography, quiet and contemplative mood.
```

---



## Ejemplo B: producto con color de marca



### Entrada del usuario

```text
Quiero un anuncio de una botella de agua deportiva azul, moderno y limpio. Color de marca naranja #FF5733.
```



### Prompt generado

```text
Premium advertising photography of a modern blue sports water bottle standing upright on a white stone pedestal. Clean minimalist composition with generous negative space around the product. Soft studio lighting creates crisp highlights on the bottle and a subtle shadow beneath it. Add small design accents in color #FF5733. Bright, energetic, premium sports-brand aesthetic.
```

---



## Ejemplo C: cartel con texto



### Entrada del usuario

```text
Haz un poster retro de un festival de música llamado Neon Nights.
```



### Prompt generado

```text
A vertical retro 1980s-inspired music festival poster. The exact headline reads: "NEON NIGHTS". The smaller text reads: "LIVE MUSIC · AUGUST 2026". A glowing neon sunset behind a dark city skyline, purple and electric blue gradients, magenta grid floor, bold geometric layout, nostalgic VHS-inspired texture, highly legible title centered at the top.
```

---



## Ejemplo D: retrato editorial



### Entrada del usuario

```text
Retrato de un chef joven en su restaurante elegante.
```



### Prompt generado

```text
Editorial portrait of a young chef standing in an elegant contemporary restaurant kitchen. He wears a crisp white chef jacket with rolled sleeves and looks confidently toward the camera. Medium shot, polished stainless-steel surfaces and warm wooden details in the background, softly blurred. Warm side lighting from the kitchen pass, realistic skin texture, shallow depth of field, high-end food magazine photography.
```

---



# 16. Instrucción final para el LLM

Al generar prompts para FLUX:

1. Escribe descripciones visuales claras y naturales.
2. Define sujeto, entorno, composición e iluminación.
3. Usa un estilo visual consistente.
4. Prioriza detalles concretos sobre adjetivos vacíos.
5. No uses *negative prompts*.
6. Pon el texto exacto entre comillas.
7. Usa códigos HEX cuando el usuario requiera colores de marca.
8. Para FLUX.2 [klein], escribe especialmente en prosa y describe bien la luz.
9. No añadas detalles importantes que cambien la intención original del usuario.
10. Entrega un prompt listo para copiar y pegar.

