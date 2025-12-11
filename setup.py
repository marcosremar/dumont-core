from setuptools import setup, find_packages

setup(
    name="dumont-core",
    version="0.2.0",
    description="Módulos compartilhados para projetos Dumont - LLM e Cloud GPU",
    author="Marcos Remar",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "httpx>=0.25.0",
        "python-dotenv>=1.0.0",
    ],
    extras_require={
        "llm": [
            "litellm>=1.0.0",
            "structlog>=23.0.0",
        ],
        "cloud": [
            "vastai>=0.1.0",
        ],
        "providers": [
            "openai>=1.0.0",
            "anthropic>=0.18.0",
            "ollama>=0.1.0",
        ],
        "all": [
            "litellm>=1.0.0",
            "structlog>=23.0.0",
            "openai>=1.0.0",
            "anthropic>=0.18.0",
            "ollama>=0.1.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "dumont-gpu=dumont_core.cloud.cli:main",
        ],
    },
)
