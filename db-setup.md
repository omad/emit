# Local Database setup

1. Install postgresql
   - install mamba-forge
     - use `~/.local/mamba` as base
     - `https://github.com/conda-forge/miniforge#mambaforge`
   - `mamba install postgresql`
   - Check that you have `~/.local/mamba/bin/pg_ctl` present

2. Fetch database from S3
   - `./scripts/db.sh fetch`

3. Start database
   - `./scripts/db.sh start`
   - check db log: `./scripts/db.sh log`

4. Configure datacube
   - `eval "$(./scripts/db.sh env)"`
   - verify
     - `datacube system check`
     - `datacube product list`
   - use `emit` Python env for the above
     - `source $HOME/.envs/emit/bin/activate`



## Appendix

### Product list

```
av3_l2a   Airborne Visible InfraRed Imaging Spectrometer (V3, L2A)
emit_l2a  EMIT L2A Estimated Surface Reflectance 60 m V001
```

### Conda info

```
>>> conda info

     active environment : base
    active env location : /home/jovyan/.local/mamba
            shell level : 1
       user config file : /home/jovyan/.condarc
 populated config files : /home/jovyan/.local/mamba/.condarc
          conda version : 23.11.0
    conda-build version : not installed
         python version : 3.10.13.final.0
                 solver : libmamba (default)
       virtual packages : __archspec=1=icelake
                          __conda=23.11.0=0
                          __glibc=2.35=0
                          __linux=5.10.217=0
                          __unix=0=0
       base environment : /home/jovyan/.local/mamba  (writable)
      conda av data dir : /home/jovyan/.local/mamba/etc/conda
  conda av metadata url : None
           channel URLs : https://conda.anaconda.org/conda-forge/linux-64
                          https://conda.anaconda.org/conda-forge/noarch
          package cache : /home/jovyan/.local/mamba/pkgs
                          /home/jovyan/.conda/pkgs
       envs directories : /home/jovyan/.local/mamba/envs
                          /home/jovyan/.conda/envs
               platform : linux-64
             user-agent : conda/23.11.0 requests/2.31.0 CPython/3.10.13 Linux/5.10.217-205.860.amzn2.x86_64 ubuntu/22.04.3 glibc/2.35 solver/libmamba conda-libmamba-solver/23.12.0 libmambapy/1.5.5
                UID:GID : 1000:100
             netrc file : None
           offline mode : False
```


