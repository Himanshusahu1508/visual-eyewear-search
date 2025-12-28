from search.search_engine import VisualSearchEngine

engine = VisualSearchEngine()

results = engine.search(
    "data/dataset_v1/raw_images/Rectangle/LenskartAir_Rectangle_07.jpg",
    top_k=5
)

for r in results:
    print(r)
