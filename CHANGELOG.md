# Changelog

## [v0.3.2](https://github.com/chipmuenk/pyfda/tree/v0.3.2) (2020-09-xx)

**Bug Fixes**
- Make compatible to matplotlib 3.3 by cleaning up hierarchy for NavigationToolbar in mpl_widgets.py
 [(Issue \#179)](https://github.com/chipmuenk/pyfda/issues/179) and get rid of mpl 3.3 related deprecation warnings. Disable zoom rectangle and pan when zoom is locked. 

- [PR \#182:](https://github.com/chipmuenk/pull/182) Get rid of deprecation warnings "Creating an ndarray from ragged nested sequences"  [(Issue \#180)](https://github.com/chipmuenk/pyfda/issues/180)
  by declaring explicitly np.array( some_ragged_list , dtype=object) or by handling the elements of ragged list indidually
  ([chipmuenk](https://github.com/chipmuenk))
  
- When the gain k has been changed, highlight the save button.

**Enhancements**

- Add cursor / annotations in plots [(Issue \#112)](https://github.com/chipmuenk/issues/112)

  Only available when [mplcursors](https://mplcursors.readthedocs.io/) module is installed and for matplotlib >= 3.1. 

- Add CHANGELOG (this file)

- Move attributions to AUTHORS.md
