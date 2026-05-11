--------------
```
     __         _       __    __           _           __
    / /_  _____(_)___  / /_  / /__      __(_)___  ___ / /
   / __ \/ ___/ / __ \/ __ \/ __/ | /| / / / __ \/ __  /
  / /_/ / /  / / /_/ / / / / /_ | |/ |/ / / / / / /_/ /
 /_.___/_/  /_/\__, /_/ /_/\__/ |__/|__/_/_/ /_/\__,_/
              /____/
 ```
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;**A Python library primarily for wind resource assessments.**

--------------

<br>

Brightwind is an open-source Python library for wind (and solar) resource analysis. It loads meteorological
timeseries data, runs common analyses — shear, long-term adjustments, correlations, distributions — and exports
results to formats used by wind analysis software such as WAsP.

📚 **Full documentation, tutorials and API reference:** https://brightwind-dev.github.io/brightwind-docs/

<br>

---
### Installation

Install brightwind into its own environment to avoid dependency clashes. Pick whichever option suits you.

#### Option 1 — venv

```bash
python -m venv brightwind_env
# Windows
brightwind_env\Scripts\activate
# macOS / Linux
source brightwind_env/bin/activate

pip install brightwind
```

#### Option 2 — conda

New to Python? We recommend [Anaconda](https://www.datacamp.com/tutorial/installing-anaconda-windows), which
bundles Python, pip and Jupyter. Then from the **Anaconda Prompt**:

```bash
conda create --name brightwind_env python=3.11
conda activate brightwind_env
pip install brightwind
```

A step-by-step Windows install walkthrough is also available in the
[tutorials](https://brightwind-dev.github.io/brightwind-docs/).

<br>

---
### Quick start

Most analysts use brightwind from a Jupyter Notebook:

```bash
pip install jupyter
jupyter notebook
```

```python
import brightwind as bw

data = bw.load_csv(bw.demo_datasets.demo_data)
bw.basic_stats(data)
```

For full examples — loading, plotting, shear, correlations, exporting — see the
[tutorials and API reference](https://brightwind-dev.github.io/brightwind-docs/).

<br>

<p>

![demo_image_1](read_me_1.png)
![demo_image_2](read_me_2.png)
</p>

<br>

---
### Contributing

Brightwind welcomes contributions from across the wind and solar industry — analysts, engineers, researchers and
developers.

- **Issues, bugs and feature requests:** [GitHub issue tracker](https://github.com/brightwind-dev/brightwind/issues)
- **Code contributions and development setup:** see [contributing.md](contributing.md)
- **General enquiries:** stephen@brightwindanalysis.com

<br>

---
### License

MIT — see [LICENSE.txt](LICENSE.txt).
