from langchain_community.document_loaders import DirectoryLoader


def main():
    data_path = "../data/AIDL"
    loader = DirectoryLoader(
        data_path,
        glob="**/*.md",
        show_progress=True,
    )
    docs = loader.load()

    print(f"loaded {len(docs)} documents")
    print(docs[0].page_content)
    print(docs[0].metadata)


if __name__ == "__main__":
    main()
