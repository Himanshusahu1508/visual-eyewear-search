import streamlit as st
import requests
from PIL import Image
import io

API_URL = "http://127.0.0.1:8000"

st.set_page_config(page_title="Visual Eyewear Search", layout="wide")

st.title("👓 Visual Eyewear Search")
st.write("Upload an image to find visually similar eyewear")

# ---------------- Sidebar Filters ----------------
st.sidebar.header("Filters")

top_k = st.sidebar.slider("Top K Results", 1, 10, 5)
brand = st.sidebar.text_input("Brand (optional)")
material = st.sidebar.text_input("Material (optional)")
price_min = st.sidebar.number_input("Min Price", min_value=0, value=0)
price_max = st.sidebar.number_input("Max Price", min_value=0, value=5000)
text_query = st.text_input("Optional text modifier (e.g. 'metal aviator')")


# ---------------- Image Upload ----------------
uploaded_file = st.file_uploader("Upload Eyewear Image", type=["jpg", "jpeg", "png", "webp"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Query Image", width=300)

    if st.button("🔍 Search"):
        with st.spinner("Searching..."):
            files = {
                "image": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)
            }

            data = {
                "top_k": top_k,
                "brand": brand,
                "material": material,
                "price_min": price_min,
                "price_max": price_max,
                "text_query": text_query
            }



            response = requests.post(f"{API_URL}/search", files=files, data=data)

            if response.status_code == 200:
                results = response.json()["results"]

                st.subheader("Results")

                cols = st.columns(len(results))

                for col, item in zip(cols, results):
                    with col:
                        image_url = f"http://127.0.0.1:8000/static/{item['image_path']}"
                        st.image(image_url, use_column_width=True)

                        st.text(f"Shape: {item['shape']}")
                        st.text(f"Brand: {item['brand']}")
                        st.text(f"Price: ₹{item['price']}")
                        st.text(f"Similarity: {item['similarity']:.2f}")

                        # Feedback button
                        if st.button(f"👍 Like {item['image_id']}"):
                            fb_data = {
                                "image_id": item["image_id"],
                                "shape": item["shape"]
                            }
                            requests.post(f"{API_URL}/feedback", data=fb_data)
                            st.success("Feedback recorded!")

            else:
                st.error("Search failed. Please try again.")
