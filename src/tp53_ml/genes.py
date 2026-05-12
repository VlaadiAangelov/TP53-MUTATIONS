"""Curated gene sets used for interpretable TP53 feature subsets and enrichment testing."""

TP53_TARGET_GENES = sorted(
    {
        "TP53",
        "CDKN1A",
        "MDM2",
        "BAX",
        "BBC3",
        "PMAIP1",
        "GADD45A",
        "RRM2B",
        "DDB2",
        "XPC",
        "FAS",
        "TNFRSF10B",
        "BTG2",
        "SESN1",
        "SESN2",
        "ZMAT3",
        "PLK3",
        "PERP",
        "FDXR",
        "PPM1D",
        "CD82",
        "SFN",
        "TIGAR",
        "AEN",
        "APAF1",
        "CASP1",
        "EI24",
        "DRAM1",
        "PANK1",
        "SUSD6",
        "TRIAP1",
        "PHLDA3",
        "PRKAB1",
    }
)

P53_PATHWAY_GENES = sorted(
    set(TP53_TARGET_GENES)
    | {
        "ATM",
        "ATR",
        "CHEK1",
        "CHEK2",
        "MDM4",
        "RB1",
        "E2F1",
        "CCND1",
        "CCNE1",
        "CDK2",
        "CDK4",
        "CDK6",
        "BCL2",
        "BCL2L1",
        "BID",
        "BAK1",
        "CASP3",
        "CASP8",
        "CASP9",
        "BRCA1",
        "BRCA2",
        "RAD51",
        "MSH2",
        "MLH1",
        "PCNA",
        "PARP1",
        "FANCD2",
        "XRCC5",
        "XRCC6",
    }
)

APOPTOSIS_GENES = sorted({
    "BAX", "BAK1", "BCL2", "BCL2L1", "BCL2L2", "MCL1", "BID", "BBC3", "PMAIP1",
    "APAF1", "CASP3", "CASP7", "CASP8", "CASP9", "CASP1", "FAS", "FADD",
    "TNFRSF10A", "TNFRSF10B", "CYCS", "XIAP", "BIRC2", "BIRC3", "BIRC5",
    "TP53", "MDM2", "PERP", "EI24", "DRAM1", "PHLDA3", "AEN", "BBC3",
})

DNA_REPAIR_GENES = sorted({
    "BRCA1", "BRCA2", "RAD51", "RAD52", "RAD54L", "XRCC1", "XRCC4", "XRCC5", "XRCC6",
    "MSH2", "MSH3", "MSH6", "MLH1", "MLH3", "PMS2", "PCNA", "PARP1", "PARP2",
    "FANCD2", "ATM", "ATR", "CHEK1", "CHEK2", "RRM2B", "DDB2", "XPC",
    "ERCC1", "ERCC2", "ERCC4", "ERCC5", "LIG1", "LIG4", "NHEJ1",
    "GADD45A", "FDXR",
})

CELL_CYCLE_GENES = sorted({
    "CCND1", "CCND2", "CCND3", "CCNE1", "CCNE2", "CCNA1", "CCNA2", "CCNB1", "CCNB2",
    "CDK1", "CDK2", "CDK4", "CDK6", "CDK7",
    "CDKN1A", "CDKN1B", "CDKN2A", "CDKN2B",
    "RB1", "E2F1", "E2F2", "E2F3", "E2F4",
    "PCNA", "GADD45A", "GADD45B", "GADD45G",
    "BTG2", "TP53", "MDM2", "PLK3", "WEE1",
    "CDC25A", "CDC25B", "CDC25C",
})

TP53_HOTSPOT_RESIDUES = {"R175", "G245", "R248", "R249", "R273", "R282"}

CANONICAL_P53_TARGETS_INFO = {
    "CDKN1A":    "cell cycle arrest (p21)",
    "MDM2":      "p53 negative regulator",
    "BAX":       "pro-apoptotic",
    "BBC3":      "pro-apoptotic (PUMA)",
    "RRM2B":     "nucleotide synthesis / DNA repair",
    "ZMAT3":     "mRNA stability / apoptosis",
    "SFN":       "14-3-3σ, cell cycle arrest",
    "TNFRSF10B": "death receptor (TRAIL-R2)",
    "GADD45A":   "DNA damage response / cell cycle",
    "DDB2":      "nucleotide excision repair",
    "FAS":       "death receptor / apoptosis",
    "SESN1":     "antioxidant / mTOR regulation",
    "PMAIP1":    "pro-apoptotic (NOXA)",
    "BTG2":      "anti-proliferative",
    "AEN":       "apoptosis enhancing nuclease",
    "FDXR":      "mitochondrial electron transport",
}

