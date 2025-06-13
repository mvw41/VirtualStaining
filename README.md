# Virtual Staining

Dieses Beispielprojekt demonstriert, wie man mit Python mehrere
neuronale Netze für unterschiedliche Färbungen aufbauen kann. Für jede
Färbung wird ein eigenes Netz (``StainNet``) verwendet. Ein weiteres Netz
(``CombineNet``) kombiniert die Vorhersagen der einzelnen Färbungen, um
fluoreszierende Aufnahmen aus Brightfield-Bildern zu erzeugen.

Die Datei ``train_cnn.py`` enthält die Definitionen der einzelnen
Stain-Netze sowie einen kleinen Trainingsablauf. Das hierbei verwendete
Kombinationsnetz ``CombineCNN`` wird in ``combine_cnn.py`` bereitgestellt.

Zusätzlich steht ein kleines ``UNet`` in ``unet.py`` bereit, das als
Basis für die Rekonstruktion einzelner Färbungen dient. Ein Beispiel ist
``train_DAPI.py``, das DAPI-Aufnahmen aus Brightfield-Bildern generiert.
Die Daten werden dabei über ``dataloader.py`` geladen, das zufällige
256×256 Tiles aus 1024×1024 Bildern ausschneidet.

Zum Starten des Beispieltrainings kann folgender Befehl verwendet
werden:

```bash
python train_cnn.py
```

Zum Trainieren des DAPI-Modells benötigt man passende Bildordner mit
Brightfield- und DAPI-Aufnahmen und ruft beispielsweise auf:

```bash
python train_DAPI.py --bf_dir /path/to/bf --dapi_dir /path/to/dapi --epochs 5
```

Die Trainingsroutinen verwenden Platzhalter- bzw. Zufallsdaten und dienen
nur zur Illustration der Netzwerkstruktur. Für reale Anwendungen müssen
passende Bilddatensätze und Verlustfunktionen eingebunden werden.
