def form_transfer(formula):
    """

    :param formula: dependent variable ~ covariant | fixed_effect |clusters'
    :type formula: str

    :return: Lists of out_col, consist_col, category_col, cluster_col, respectively.
    """
    phenotype = formula.replace(' ', '').split("~")
    assert len(phenotype) == 2, f"Formula must have a phenotype separated by ~ yet failed to find via splitting " \
                                f"for {formula}"

    # Split the right hand side out, validate there is only one phenotype
    phenotype, rhs = phenotype
    assert len(phenotype.split("+")) == 1, f"Can only provide a single phenotype per OLS yet found " \
                                           f"{phenotype.split('+')}"

    segments = rhs.split("|")
    assert 1 <= len(segments) < 4 and sum([len(seg) for seg in segments]) != 0, \
        f"Right hand side should be 'covariant | fixed_effect | cluster's meaning.\nAll models should have at least " \
        f"a covariant, and can be max length of 3 yet found: {segments}"

    # Ensure segments are of length 3
    segments = segments + ["" for _ in range(3 - len(segments))]

    # Split each section on + then return the phenotype, covariant, fixed_effect and cluster variables
    sections = [[phenotype]] + [section.split("+") if len(section) > 0 else [] for section in segments]
    phenotype, covariant, fixed_effects, clusters = sections
    return phenotype, covariant, fixed_effects, clusters
