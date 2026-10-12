# MCNS data credits

`mcns_subnetwork.npz` and `mcns_subnetwork.json` are adapted from the Male CNS
v1.0 connectome, obtained as MCNS CSV exports through FlyWire Codex.

The upstream Male CNS project is a collaboration between FlyEM at HHMI Janelia,
the University of Cambridge Department of Zoology, the MRC Laboratory of
Molecular Biology, and Google Research. Its official download page identifies
the dataset license as **Creative Commons Attribution 4.0 International**.

- [Male CNS project](https://male-cns.janelia.org/)
- [Official downloads and license](https://male-cns.janelia.org/download/)
- [CC BY 4.0 terms](https://creativecommons.org/licenses/by/4.0/)
- [FlyWire Codex MCNS explorer](https://codex.flywire.ai/?dataset=mcns)

I selected 256 neurons and the 3,326 directed connections between them, filtered
individual neuropil rows below five synapses, aggregated counts, converted
counts to log weights, applied a simplified neurotransmitter sign rule,
normalized incoming weights, and added artificial Celeste sensor and action
mappings. Learned gains are project model
parameters, not measured biological changes. The supplied JSON records the
selection method and original export checksums.

I retain the data's upstream terms separately from my code's MIT license.
I am independent of the dataset creators and do not imply their endorsement.
I do not include the original full CSV exports.
