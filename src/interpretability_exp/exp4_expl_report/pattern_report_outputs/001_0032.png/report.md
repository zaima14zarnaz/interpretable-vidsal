# Visual pattern report

Video 001 · 0032.png · frame position 32 · complete

![Selected frame and salient anchors](salient_anchors.png)

![Predicted saliency overlay on the original frame](predicted_saliency.png)

Blue = lower predicted saliency; red = higher predicted saliency. Colors are normalized within each frame.

## Anchor 30: cell (2, 6)

Video 001 · frame 32 (0032.png)

![Anchor](crops/anchor_30.png)

```json
{
  "Description of each patch": [
    {
      "Anchor": {
        "Low-level visual features": "Red and skin-toned surfaces with soft textures.",
        "Semantic object or part": "A person wearing a red shirt."
      }
    },
    {
      "Match 1": {
        "Low-level visual features": "Dark grey shape against a white background.",
        "Semantic object or part": "An animal's tail or shadow."
      }
    },
    {
      "Match 2": {
        "Low-level visual features": "Beige and white vertical linear folds.",
        "Semantic object or part": "Fabric pleats or draped material."
      }
    },
    {
      "Match 3": {
        "Low-level visual features": "Green and dark mottled textures.",
        "Semantic object or part": "Foliage or trees."
      }
    },
    {
      "Match 4": {
        "Low-level visual features": "Green and yellowish-brown patchy surface.",
        "Semantic object or part": "Grass or terrain."
      }
    },
    {
      "Match 5": {
        "Low-level visual features": "Pink surface with a white graphic mark.",
        "Semantic object or part": "A pink t-shirt with a design."
      }
    },
    {
      "Match 6": {
        "Low-level visual features": "Light beige upper region and a dark horizontal band below.",
        "Semantic object or part": "A wall or surface edge."
      }
    },
    {
      "Match 7": {
        "Low-level visual features": "Dark vertical background with yellow text or graphics.",
        "Semantic object or part": "Product packaging or printed text."
      }
    },
    {
      "Match 8": {
        "Low-level visual features": "Mottled green, yellow, and white patches.",
        "Semantic object or part": "A landscape or water surface."
      }
    },
    {
      "Match 9": {
        "Low-level visual features": "High-contrast black, white, and dark tones with linear contours.",
        "Semantic object or part": "An illustrated face or hair."
      }
    },
    {
      "Match 10": {
        "Low-level visual features": "Dark bluish-gray curved surface with subtle highlights.",
        "Semantic object or part": "A helmet or visor."
      }
    }
  ],
  "Description of the common visual pattern": "There is no clear visual pattern or semantic theme shared by a majority of the patches, as each image depicts completely different objects such as clothing, animals, foliage, text, and landscapes.",
  "Visual pattern title": "None",
  "Supporting patches": [],
  "Non-supporting patches": [
    "Anchor",
    "Match 1",
    "Match 2",
    "Match 3",
    "Match 4",
    "Match 5",
    "Match 6",
    "Match 7",
    "Match 8",
    "Match 9",
    "Match 10"
  ]
}
```

Anchor top-k prototypes: [31, 91, 104]

Prototypes common to every retrieved match: [3]

| Match number | Node | Video / frame | Saliency label | Percentile | Stage-4 cosine |
| --- | --- | --- | --- | --- | --- |
| 1 | 116971 | 289 / 0288.png | non-salient | 0.964 | 0.3687 |
| 2 | 2319 | 006 / 0288.png | salient | 0.976 | 0.1522 |
| 3 | 1940 | 005 / 0288.png | non-salient | 0.880 | 0.2815 |
| 4 | 1953 | 005 / 0288.png | non-salient | 0.952 | 0.2907 |
| 5 | 145870 | 362 / 0544.png | non-salient | 0.952 | 0.2208 |
| 6 | 135438 | 335 / 0160.png | salient | 1.000 | 0.2697 |
| 7 | 225343 | 563 / 0160.png | non-salient | 0.928 | 0.2153 |
| 8 | 120833 | 299 / 0160.png | salient | 0.976 | 0.1713 |
| 9 | 87558 | 212 / 0544.png | non-salient | 0.940 | 0.1538 |
| 10 | 86057 | 209 / 0416.png | salient | 0.976 | 0.1565 |

Node 116971 — shared with anchor: [31]; recurring across matches (ID/count): [{'id': 3, 'count': 10}, {'id': 31, 'count': 6}, {'id': 51, 'count': 9}]

Node 2319 — shared with anchor: [31]; recurring across matches (ID/count): [{'id': 3, 'count': 10}, {'id': 31, 'count': 6}, {'id': 51, 'count': 9}]

Node 1940 — shared with anchor: []; recurring across matches (ID/count): [{'id': 3, 'count': 10}, {'id': 51, 'count': 9}, {'id': 81, 'count': 4}]

Node 1953 — shared with anchor: []; recurring across matches (ID/count): [{'id': 3, 'count': 10}, {'id': 51, 'count': 9}, {'id': 81, 'count': 4}]

Node 145870 — shared with anchor: [31]; recurring across matches (ID/count): [{'id': 3, 'count': 10}, {'id': 31, 'count': 6}, {'id': 51, 'count': 9}]

Node 135438 — shared with anchor: [31]; recurring across matches (ID/count): [{'id': 3, 'count': 10}, {'id': 31, 'count': 6}, {'id': 51, 'count': 9}]

Node 225343 — shared with anchor: []; recurring across matches (ID/count): [{'id': 3, 'count': 10}, {'id': 81, 'count': 4}]

Node 120833 — shared with anchor: []; recurring across matches (ID/count): [{'id': 3, 'count': 10}, {'id': 51, 'count': 9}, {'id': 81, 'count': 4}]

Node 87558 — shared with anchor: [31]; recurring across matches (ID/count): [{'id': 3, 'count': 10}, {'id': 31, 'count': 6}, {'id': 51, 'count': 9}]

Node 86057 — shared with anchor: [31]; recurring across matches (ID/count): [{'id': 3, 'count': 10}, {'id': 31, 'count': 6}, {'id': 51, 'count': 9}]

## Anchor 29: cell (2, 5)

Video 001 · frame 32 (0032.png)

![Anchor](crops/anchor_29.png)

```json
{
  "Description of each patch": [
    {
      "Anchor": {
        "Low-level visual features": "Dark color tones, vertical orientation, blurry texture",
        "Semantic object or part": "Dark hair and shoulder of a person near water"
      }
    },
    {
      "Match 1": {
        "Low-level visual features": "Vertical alternating dark and light stripes, linear edges",
        "Semantic object or part": "Striped fabric or clothing"
      }
    },
    {
      "Match 2": {
        "Low-level visual features": "Dark blue and skin-toned regions with soft lighting",
        "Semantic object or part": "Shoulder, neck, and dark clothing"
      }
    },
    {
      "Match 3": {
        "Low-level visual features": "Vertical alternating dark and light stripes, linear edges",
        "Semantic object or part": "Striped fabric or clothing"
      }
    },
    {
      "Match 4": {
        "Low-level visual features": "Dark brown and white patches, vertical leg-like shapes",
        "Semantic object or part": "Lower legs and body of a horse"
      }
    },
    {
      "Match 5": {
        "Low-level visual features": "Dark wings, white central body, green background",
        "Semantic object or part": "Flying or standing bird"
      }
    },
    {
      "Match 6": {
        "Low-level visual features": "Muted gray-brown textured background with faint white letters",
        "Semantic object or part": "Ground surface with text overlay"
      }
    },
    {
      "Match 7": {
        "Low-level visual features": "Dark limb silhouette against blue and white horizontal lines",
        "Semantic object or part": "Leg near a court line"
      }
    },
    {
      "Match 8": {
        "Low-level visual features": "Tan and dark brown fur texture with a small highlighted snout area",
        "Semantic object or part": "Animal muzzle or nose"
      }
    },
    {
      "Match 9": {
        "Low-level visual features": "Dark solid fur texture, white band near the bottom",
        "Semantic object or part": "Horse body and leg"
      }
    },
    {
      "Match 10": {
        "Low-level visual features": "Purple surface with a white line and a blurry leg",
        "Semantic object or part": "Tennis court surface and player's leg"
      }
    }
  ],
  "Description of the common visual pattern": "There is no majority visual pattern or shared semantic category among the patches, as each patch depicts completely unrelated objects and scenes such as clothing, animals, sports courts, and text.",
  "Visual pattern title": "None",
  "Supporting patches": [],
  "Non-supporting patches": [
    "Anchor",
    "Match 1",
    "Match 2",
    "Match 3",
    "Match 4",
    "Match 5",
    "Match 6",
    "Match 7",
    "Match 8",
    "Match 9",
    "Match 10"
  ]
}
```

Anchor top-k prototypes: [31, 91, 126]

Prototypes common to every retrieved match: [126]

| Match number | Node | Video / frame | Saliency label | Percentile | Stage-4 cosine |
| --- | --- | --- | --- | --- | --- |
| 1 | 12210 | 029 / 0928.png | non-salient | 0.964 | 0.1387 |
| 2 | 174993 | 435 / 0032.png | non-salient | 0.940 | 0.0545 |
| 3 | 12209 | 029 / 0928.png | salient | 0.976 | 0.1606 |
| 4 | 67658 | 161 / 0160.png | salient | 0.976 | 0.1709 |
| 5 | 118791 | 294 / 0160.png | non-salient | 0.892 | 0.3344 |
| 6 | 103226 | 253 / 0416.png | non-salient | 0.627 | 0.1220 |
| 7 | 233324 | 582 / 0032.png | non-salient | 0.952 | 0.2015 |
| 8 | 173429 | 431 / 0032.png | non-salient | 0.940 | 0.1953 |
| 9 | 67657 | 161 / 0160.png | non-salient | 0.964 | 0.1932 |
| 10 | 48941 | 115 / 0416.png | non-salient | 0.952 | 0.1159 |

Node 12210 — shared with anchor: [31, 126]; recurring across matches (ID/count): [{'id': 31, 'count': 4}, {'id': 65, 'count': 2}, {'id': 126, 'count': 10}]

Node 174993 — shared with anchor: [126]; recurring across matches (ID/count): [{'id': 68, 'count': 3}, {'id': 90, 'count': 6}, {'id': 126, 'count': 10}]

Node 12209 — shared with anchor: [31, 126]; recurring across matches (ID/count): [{'id': 31, 'count': 4}, {'id': 65, 'count': 2}, {'id': 126, 'count': 10}]

Node 67658 — shared with anchor: [91, 126]; recurring across matches (ID/count): [{'id': 91, 'count': 2}, {'id': 104, 'count': 2}, {'id': 126, 'count': 10}]

Node 118791 — shared with anchor: [126]; recurring across matches (ID/count): [{'id': 90, 'count': 6}, {'id': 126, 'count': 10}]

Node 103226 — shared with anchor: [126]; recurring across matches (ID/count): [{'id': 90, 'count': 6}, {'id': 104, 'count': 2}, {'id': 126, 'count': 10}]

Node 233324 — shared with anchor: [91, 126]; recurring across matches (ID/count): [{'id': 90, 'count': 6}, {'id': 91, 'count': 2}, {'id': 126, 'count': 10}]

Node 173429 — shared with anchor: [31, 126]; recurring across matches (ID/count): [{'id': 31, 'count': 4}, {'id': 90, 'count': 6}, {'id': 126, 'count': 10}]

Node 67657 — shared with anchor: [31, 126]; recurring across matches (ID/count): [{'id': 31, 'count': 4}, {'id': 68, 'count': 3}, {'id': 126, 'count': 10}]

Node 48941 — shared with anchor: [126]; recurring across matches (ID/count): [{'id': 68, 'count': 3}, {'id': 90, 'count': 6}, {'id': 126, 'count': 10}]

## Anchor 18: cell (1, 6)

Video 001 · frame 32 (0032.png)

![Anchor](crops/anchor_18.png)

```json
{
  "Description of each patch": [
    {
      "Anchor": {
        "Low-level visual features": "Flesh-toned and dark regions with blurry vertical and horizontal gradients.",
        "Semantic object or part": "Upper body and torso of a person."
      }
    },
    {
      "Match 1": {
        "Low-level visual features": "Dark diagonal lines intersecting with a bright white highlight area.",
        "Semantic object or part": "Structural lines or railings."
      }
    },
    {
      "Match 2": {
        "Low-level visual features": "Horizontal blue and dark bands with vertical rectangular windows or port holes.",
        "Semantic object or part": "Side of a ship or building facade."
      }
    },
    {
      "Match 3": {
        "Low-level visual features": "Angular light-colored surfaces meeting in stepped planes under shadow.",
        "Semantic object or part": "Building roofs or architectural cornices."
      }
    },
    {
      "Match 4": {
        "Low-level visual features": "Smooth flesh-toned gradients with a subtle vertical contour.",
        "Semantic object or part": "Skin of a person's body or limb."
      }
    },
    {
      "Match 5": {
        "Low-level visual features": "Dense dark and light speckles forming a mottled textured field.",
        "Semantic object or part": "Crowd of people in a stadium."
      }
    },
    {
      "Match 6": {
        "Low-level visual features": "Horizontal blocks of turquoise, white, and dark blue with faint reflections.",
        "Semantic object or part": "Swimming pool edge and water."
      }
    },
    {
      "Match 7": {
        "Low-level visual features": "Brownish-tan mottled texture with a central dark irregular blob.",
        "Semantic object or part": "Earthy ground or a small animal/creature."
      }
    },
    {
      "Match 8": {
        "Low-level visual features": "Brownish-tan mottled texture with a central dark irregular blob.",
        "Semantic object or part": "Earthy ground or a small animal/creature."
      }
    },
    {
      "Match 9": {
        "Low-level visual features": "Grey speckled grainy texture with a bright turquoise curve at the bottom corner.",
        "Semantic object or part": "Textured surface with a colored edge."
      }
    },
    {
      "Match 10": {
        "Low-level visual features": "Light blue background area above a rounded green-brown object and dark vertical shapes.",
        "Semantic object or part": "Object against an open sky or light background."
      }
    }
  ],
  "Description of the common visual pattern": "There is no clear visual pattern shared by most of the target patches. Each patch depicts entirely different semantic objects, colors, and textures (e.g., human skin, ship siding, architectural roofs, pool water, earthy ground, and crowd textures), failing to reach a majority agreement.",
  "Visual pattern title": "None",
  "Supporting patches": [],
  "Non-supporting patches": [
    "Anchor",
    "Match 1",
    "Match 2",
    "Match 3",
    "Match 4",
    "Match 5",
    "Match 6",
    "Match 7",
    "Match 8",
    "Match 9",
    "Match 10"
  ]
}
```

Anchor top-k prototypes: [3, 57, 81]

Prototypes common to every retrieved match: [51]

| Match number | Node | Video / frame | Saliency label | Percentile | Stage-4 cosine |
| --- | --- | --- | --- | --- | --- |
| 1 | 130556 | 322 / 0672.png | non-salient | 0.831 | 0.0904 |
| 2 | 124866 | 310 / 0288.png | non-salient | 0.940 | 0.1968 |
| 3 | 126224 | 313 / 0288.png | non-salient | 0.771 | 0.1312 |
| 4 | 33557 | 076 / 0672.png | non-salient | 0.952 | 0.2648 |
| 5 | 22699 | 053 / 0800.png | non-salient | 0.807 | 0.2682 |
| 6 | 16193 | 039 / 0160.png | non-salient | 0.964 | 0.1097 |
| 7 | 34736 | 079 / 0544.png | non-salient | 0.916 | 0.1416 |
| 8 | 34652 | 079 / 0416.png | non-salient | 0.867 | 0.1908 |
| 9 | 218919 | 546 / 0544.png | non-salient | 0.904 | 0.1233 |
| 10 | 80012 | 192 / 0160.png | non-salient | 0.867 | 0.1993 |

Node 130556 — shared with anchor: []; recurring across matches (ID/count): [{'id': 51, 'count': 10}, {'id': 115, 'count': 8}]

Node 124866 — shared with anchor: []; recurring across matches (ID/count): [{'id': 31, 'count': 2}, {'id': 51, 'count': 10}, {'id': 115, 'count': 8}]

Node 126224 — shared with anchor: []; recurring across matches (ID/count): [{'id': 51, 'count': 10}]

Node 33557 — shared with anchor: [81]; recurring across matches (ID/count): [{'id': 51, 'count': 10}, {'id': 81, 'count': 2}, {'id': 115, 'count': 8}]

Node 22699 — shared with anchor: []; recurring across matches (ID/count): [{'id': 31, 'count': 2}, {'id': 51, 'count': 10}, {'id': 115, 'count': 8}]

Node 16193 — shared with anchor: [3, 81]; recurring across matches (ID/count): [{'id': 3, 'count': 5}, {'id': 51, 'count': 10}, {'id': 81, 'count': 2}]

Node 34736 — shared with anchor: [3]; recurring across matches (ID/count): [{'id': 3, 'count': 5}, {'id': 51, 'count': 10}, {'id': 115, 'count': 8}]

Node 34652 — shared with anchor: [3]; recurring across matches (ID/count): [{'id': 3, 'count': 5}, {'id': 51, 'count': 10}, {'id': 115, 'count': 8}]

Node 218919 — shared with anchor: [3]; recurring across matches (ID/count): [{'id': 3, 'count': 5}, {'id': 51, 'count': 10}, {'id': 115, 'count': 8}]

Node 80012 — shared with anchor: [3]; recurring across matches (ID/count): [{'id': 3, 'count': 5}, {'id': 51, 'count': 10}, {'id': 115, 'count': 8}]
