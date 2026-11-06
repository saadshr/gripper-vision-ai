"""La phase courante autorise la prédiction et les tests, pas l'entraînement."""
def refuser_action(action: str) -> None:
    """Bloque une action sensible jusqu'à une nouvelle autorisation du propriétaire."""
    raise RuntimeError(
        f"Action {action!r} bloquée : installation, préparation et entraînement "
        "nécessitent une nouvelle autorisation. La prédiction reste disponible."
    )
