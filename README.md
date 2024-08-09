Pour la Reconnaissance d'entités nommées utiliser le script Use_spacy.py,

Le script Use_spacy.py prend en entrée du texte (voir l'architecture de dossier dans "DATA-echantillon"),

Le script Use_spacy.py propose une sortie dictionnaire au format json + une sortie type csv .bio (format IOB2),

Le chemin par défaut pour utiliser Le script Use_spacy.py est ../DATA il peut être changé avec l'option -d,

Dans le répertoire "DATA-echantillon" figure la manière dont le dossier DATA doit être architecturé.

## Nomenclature

### Nommage des fichiers bio

- **tabO**: ajouter un O après une tabulation  qui n'est suivie de rien
- **tabR**: enlever les tabulations quand elles ne sont pas suivies d'une annotation
- **tabS**: enlever les tabulation quoi qu'il arrive
