from langchain_community.document_loaders import DirectoryLoader


def main():
    data_path = "../../Personal-Cheat-Sheets"
    loader = DirectoryLoader(
        data_path,
        glob="**/*.md",
        show_progress=True,
    )
    docs = loader.load()

    print(f"loaded {len(docs)} documents")


if __name__ == "__main__":
    main()
