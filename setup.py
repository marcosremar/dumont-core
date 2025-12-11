from setuptools import setup, find_packages

setup(
    name="dumont-shared",
    version="0.1.0",
    description="Módulos compartilhados para projetos Dumont",
    author="Marcos Remar",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "httpx>=0.25.0",
        "python-dotenv>=1.0.0",
    ],
    extras_require={
        "openrouter": ["litellm>=1.0.0"],
        "ollama": ["ollama>=0.1.0"],
    },
)
