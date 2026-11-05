"""Prepare a site's own input without assigning variants to other sites."""
from .model import GenePlan
from .output import write_outputs
from .pipeline import build_blocks, extract_genotypes


def prepare_site_blocks(*, pgen_prefix, gene_variants, rows, out_dir, extractor, party,
                        keep_zero_variants=False):
    plans = tuple(GenePlan(group.gene, tuple((variant, party) for variant in group.variants))
                  for group in gene_variants)
    extracted, genotypes, columns = extract_genotypes(pgen_prefix, rows, plans, extractor)
    blocks = build_blocks(extracted, genotypes, columns, keep_zero_variants)
    write_outputs(out_dir, blocks, rows, party=party)
    return out_dir
