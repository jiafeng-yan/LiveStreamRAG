from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as f:
    requirements = f.read().splitlines()

setup(
    name="llm_rag",
    version="0.1.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="基于大型语言模型的检索增强生成系统",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/llm_rag",
    packages=find_packages(),
    install_requires=requirements,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    entry_points={
        "console_scripts": [
            "llm_rag=main:main",
        ],
    },
    include_package_data=True,
) 