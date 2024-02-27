========================
EMIT Data loading driver
========================

Python environment Setup
------------------------

Obtain earth-data access token from `earthdata`_. Save it to ``earth-data.tk``.

.. code-block:: bash

    echo "Paste token below, then press Ctrl-D"
    cat | install -m 600 /dev/stdin earth-data.tk

Run script:

.. code-block:: bash

    env EARTHDATA_TOKEN="$(cat earth-data.tk)" ./scripts/setup-py-env.sh

This will:

1. Create ``emit`` environment in ``~/.envs/emit``
2. Register with jupyter (with extra environment variables)
3. Install packages

Data Preparation
----------------

- ``emit-stac.zip`` STAC items generated from ``.cmr.json`` and ``.nc.dmrpp``
- Generate this using `collect-stac-md`_ notebook 

Run Example Notebook
--------------------

- Open `hs-emit-example`_ notebook in jupyter

.. _earthdata: https://urs.earthdata.nasa.gov/
.. _collect-stac-md: notebooks/collect-stac-md.ipynb
.. _hs-emit-example: notebooks/hs-emit-example.ipynb

