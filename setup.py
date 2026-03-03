from setuptools import setup, find_packages

setup(
    name="invasive_plant_identifier",
    version="0.1.0",
    description="Plant species identifier using PyTorch and Streamlit",
    author="Your Name",
    packages=find_packages(),
    install_requires=[
        "torch>=1.13",
        "torchvision>=0.14",
        "streamlit>=1.0",
        "pandas>=1.0",
        "opencv-python>=4.0",
        "folium>=0.12",
        "pydeck>=0.8",
        "pillow>=8.0",
        "pytest>=6.0",
    ],
    python_requires=">=3.8",
)
