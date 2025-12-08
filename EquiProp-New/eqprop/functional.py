
def compute_betas(variant: str, beta: float):
    if variant == 'centered':
        return -beta, +beta, 2 * beta
    if variant == 'positive':
        return 0.0, +beta, beta
    if variant == 'negative':
        return -beta, 0.0, beta
    raise ValueError(f"Unknown variant: {variant}")
