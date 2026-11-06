"""Exécution réelle du script Streamlit via AppTest, sans serveur ni entraînement."""
import unittest
from sklearn import __version__ as sklearn_version
from streamlit.testing.v1 import AppTest
from src.config import ROOT_DIR

class TestStreamlit(unittest.TestCase):
    def setUp(self):
        self.app = AppTest.from_file(str(ROOT_DIR / "app.py"), default_timeout=30).run()
        self.assertEqual(len(self.app.exception), 0)

    def test_navigation_and_historical_values(self):
        self.app.sidebar.radio[0].set_value("Données et résultats").run()
        self.assertEqual(len(self.app.exception), 0)
        self.assertEqual([m.value for m in self.app.metric], ["32%", "33%"])
        self.app.sidebar.radio[0].set_value("À propos").run()
        self.assertEqual(len(self.app.exception), 0)

    def test_dataset_prediction_runs_without_training(self):
        source = next(r for r in self.app.radio if r.label == "Source de la photo")
        source.set_value("Essayer une photo du dataset").run()
        self.assertEqual(len(self.app.selectbox), 1)
        self.app.button[0].click().run()
        self.assertEqual(len(self.app.exception), 0)
        self.assertEqual(len(self.app.error), 0)
        self.assertEqual(len(self.app.metric), 1)
        if sklearn_version != "1.8.0":
            self.assertTrue(any("version installée" in w.value for w in self.app.warning))

if __name__ == "__main__":
    unittest.main()
