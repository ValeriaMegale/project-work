### Il Motore Euristico (Core Solver)

Queste sono le funzioni che guidano la costruzione del percorso e la logica di unione.

* **`merge_solver(problem, max_iter, time_limit, neighbor_count, seed)`**
    * **Cosa fa:** È l'interfaccia pubblica (entry point) del tuo risolutore, strutturata per rispettare le specifiche del progetto. Chiama il motore di ottimizzazione interno e, una volta ottenuto il percorso finale ricostruito, esegue un ricalcolo rigoroso ed esatto del costo totale, validando le distanze tramite `nx.shortest_path_length` e applicando la formula della penalità sul peso. Restituisce la tupla `(percorso, costo)`.

* **`_core_heuristic_optimization(problem, max_iterations, time_lim, k_neighbors, random_seed)`**
    * **Cosa fa:** È il vero "cervello" dell'algoritmo (Incremental Merge Heuristic). Inizializza una soluzione in cui ogni città ha un suo viaggio dedicato dal deposito. Poi, precalcola le distanze (con Dijkstra) e i *K* vicini più prossimi per ogni nodo. In un ciclo (limitato da tempo e iterazioni), valuta costantemente se l'unione di due percorsi vicini porta a un risparmio matematico (*savings*), applicando un leggero rumore casuale per evitare ottimi locali. Infine, si occupa di ricostruire un percorso continuo e valido per il grafo, gestendo nativamente la presenza di archi mancanti.

* **`evaluate_trip_cost(sequence)`** *(Funzione annidata dentro core_heuristic)*
    * **Cosa fa:** Calcola in tempo reale il costo di una specifica sottosequenza di nodi (partendo e tornando al deposito). Usa la matrice delle distanze precalcolata e applica la formula del costo super-lineare, simulando l'accumulo del peso a ogni tappa. È la funzione che permette di capire se un'unione di due viaggi è effettivamente vantaggiosa.

---

### Il Post-Processamento Analitico (Beta Optimizer)

Queste funzioni si occupano del raffinamento matematico una volta che la topologia (l'ordine di visita) è stata decisa.

* **`path_optimizer(path, problem)`**
    * **Cosa fa:** Implementa la scomposizione analitica dei viaggi. Prende in input un percorso continuo e, se il parametro β è maggiore di 1 (penalità super-lineare), calcola il numero ottimo *N\** di viaggi in cui conviene suddividere il carico. Restituisce un nuovo percorso in cui le stesse tappe vengono ripetute *N_opt* volte, trasportando una frazione del carico originale, minimizzando così l'impatto esplosivo del peso sulle distanze.

---

### Moduli di Supporto

A seconda di come hai pacchettizzato il codice finale, potresti dover menzionare queste funzioni:

* **`check_feasibility(walk, instance)`** (se usi un modulo di validazione separato)
    * **Cosa fa:** Esegue un controllo di integrità sul percorso finale prima della consegna. Verifica che il percorso inizi e finisca al deposito, che attraversi solo archi realmente esistenti nel grafo (permettendo la consegna anche su grafi sparsi) e che l'oro raccolto coincida perfettamente con quello richiesto entro una soglia di tolleranza, garantendo l'ammissibilità della soluzione.

---
