import json
import os

sample = [
    {
        "artifact_id": "artifact_001",
        "name": "Bust of Nefertiti",
        "artifact_type": "statue",
        "museum": "Neues Museum",
        "era": "New Kingdom",
        "dynasty": "18th Dynasty",
        "material": "Limestone and stucco",
        "location": "Berlin, Germany",
        "description_en": "A painted stucco-coated limestone bust of Queen Nefertiti.",
        "description_ar": "تمثال نصفي ملون للملكة نفرتيتي مصنوع من الحجر الجيري ومغطى بالجص.",
        "story_en": "This bust is one of the most famous images of ancient Egypt.",
        "story_ar": "يعد هذا التمثال النصفي من أشهر صور مصر القديمة.",
        "audio_script_en": "You are looking at the Bust of Nefertiti, one of the best-known works of ancient Egyptian art.",
        "audio_script_ar": "أنت تنظر الآن إلى تمثال نفرتيتي النصفي، وهو من أشهر أعمال الفن المصري القديم."
    }
]

os.makedirs("app/data/artifacts", exist_ok=True)
with open("app/data/artifacts/metadata.json", "w", encoding="utf-8") as f:
    json.dump(sample, f, ensure_ascii=False, indent=2)

print("metadata.json created.")