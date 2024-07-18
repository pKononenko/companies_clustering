# Companies Clustering

Move all companies `.txt` files inside `companies_data` folder.

Command to build a Cython modules (MANDATORY):

```

python setup.py build_ext --inplace

```

Example command to convert data to vector DB with embeddings:

```

python cli.py -o 1 --num_docs 20

```

Example command to search for similar companies:

```

python cli.py -o 2 --new_doc "company.txt"

```

CLI options:

* `-o` stands for cli option. `1` - vectorizing data. `2` - search for similar companies.
* `--num_docs` stands for number of documents to save into vector database.
* `--new_doc` stands for target document we use for similar documents search.
* `--top_k` stands for number of similar documents we searching for.
