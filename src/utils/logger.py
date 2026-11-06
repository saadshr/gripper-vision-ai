"""Journalisation standard sans création de fichier ni dossier à l'import."""
import logging

def get_logger(name: str, *, level: int = logging.INFO) -> logging.Logger:
    """Retourne un logger silencieux par défaut ; l'appelant peut configurer la console."""
    logger = logging.getLogger(name)
    logger.setLevel(level)
    if not logger.handlers:
        logger.addHandler(logging.NullHandler())
    return logger
