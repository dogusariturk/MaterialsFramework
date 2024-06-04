from setuptools import setup

if __name__ == "__main__":
    try:
        setup()
    except:  # noqa
        print(
            "\n\nAn error occurred while building the project, "
            "please ensure you have the most updated version of setuptools, "
            "and wheel with:\n"
            "   pip install -U setuptools wheel\n\n"
        )
        raise
