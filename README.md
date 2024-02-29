# Sloppy Search
Sloppily search through your vectors! Aiming to be quick. This is an in memory
"database" which stores vectors for the sole purpose of similarity search.

## Installation
`pip install git+https://github.com/mustafa-tariqk/SloppySearch`

## Usage
First, import the `sloppy_search` module:

```python
from src import sloppy_search as ss
import numpy as np
```

Initialize the database:
```python
DB = ss.Database()
```

I like to define the size, number of vectors, dimensions, and results:
```python
SIZE = np.int32 # num size
VECTORS = 2 ** 22 # ~5M, nums vectors
DIMENSION = 2 ** 9 # ~500, dimension of vector
RESULTS = 2 ** 2 # ~5, this is top k to return
```

Generate a query vector:
```py
query_vector = np.random.uniform(low=np.iinfo(SIZE).min,
                                 high=np.iinfo(SIZE).max, 
                                 size=DIMENSION)
```

Database population (can also be done 1 vector at a time)
```py
vectors = np.random.uniform(low=np.iinfo(SIZE).min, 
                            high=np.iinfo(SIZE).max, 
                            size=(VECTORS, DIMENSION))
DB.add_vectors(vectors)
```

And finally the search
```py
distances, indices = DB.search(query_vector, RESULTS)
```