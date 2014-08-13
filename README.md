isis-praktikum
==============

Dieses script führt zwei Topic-Modellierungen (LSA und LDA) über eine verfügbare Textsammlung durch.
Danach können Referenztexte, die z.B. aus einem Export einer MaxQDA-Analyse bestehen, abgefragt werden und die Aussagekraft des Topic-Modells getestet werden.

Dieses script entstand im Rahmen eines Praktikums am Institut für Systemwissenschaften, Innovations- und Nachhaltigkeitsforschung der Universität Graz (Juni 2014).


How-To
------

Voraussetzungen:
- python 2.6
- gensim
 

Inputs:
- ein Ordner "files" mit den Artikeln in *.pdf-Format
- ein Ordner "testdata" mit dem MaxQDA-Export sowie einer Zuordnung zwischen Topic-Nummern

Parameter:
- Topic-zahl

Ausführung:
- per Kommandozeile in den entsprechenden Ordner wechseln
- python sustainabilitylsa.py
- warten
- Ergebnisse werden in diesem Ordner abgelegt

Outputs:
- Topic-Liste
- Tabelle Konfusionswerte