# Plan badania: wpływ technik promptowania (rozumowanie i kontekst)

- **cel badania:** Sprawdzenie, jak warianty promptowania `LLMPlayer` wpływają na jakość gry w Wilka i Owce w porównaniu z punktem odniesienia. Badamy dwa czynniki: sposób odpowiedzi (bezpośrednia odpowiedź vs Chain of Thought, czyli opisowa analiza zagrożeń przed wyborem ruchu) oraz kontekst (niezależne prompty vs jedna rozmowa z historią na partię). Wynik pokaże, czy warto płacić kosztem tokenów i czasu za jawne rozumowanie i za pamięć rozmowy w zadaniach wymagających planowania przestrzennego.

- **hipotezy badawcze:**
  - H1: wymuszenie analizy przed ruchem (CoT) znacząco zwiększy odsetek wygranych partii przez LLM w porównaniu do bezpośredniej odpowiedzi.
  - H2: prowadzenie partii w jednej rozmowie z historią znacząco zwiększy odsetek wygranych w porównaniu do niezależnych promptów.
  - H3: efekt CoT zależy od trybu kontekstu (interakcja czynników).

- **metoda:**
  - **jakie narzędzie badawcze:** model symulacyjny - własna implementacja gry (1 wilk, 4 owce, bez bicia) oraz rozgrywki LLM vs przeciwnik sterowany algorytmem .
  - **jakie eksperymenty, warunki / założenia eksperymentu:**
    - Role: LLM jako wilk i LLM jako owce.
    - Układ czynnikowy 2x2, czyli 4 warianty `LLMPlayer`: {bezpośrednia odpowiedź, CoT} x {bez stanu, rozmowa}.
    - Rozumowanie: w obu wariantach model zwraca tylko numer ruchu z listy legalnych ruchów [1..n]. W CoT najpierw osobny prompt generuje analizę zagrożeń, a drugi prompt dostaje tę analizę i wybiera ruch; w odpowiedzi bezpośredniej jest tylko prompt wyboru ruchu.
    - Kontekst: bez stanu każdy prompt jest niezależny (zasady + aktualny stan gry); w rozmowie jedna partia = jeden „czat” z LLM-em, każda nowa wiadomość zawiera pełny aktualny stan i ostatni ruch przeciwnika, a historia rośnie z każdym ruchem (bez przycinania, ograniczona limitem tur).
    - Wspólne dla wszystkich wariantów: te same zasady w prompcie, dualna reprezentacja planszy (współrzędne + siatka ASCII), wstrzyknięta lista legalnych ruchów.
    - Przeciwnik: mamy kilka opcji i zdecydujemy, którą wybrać, gdy zobaczymy, jak radzą sobie LLM-y. Opcje: (a) minimax o głębokości 2-4 z podzbiorem heurystyk, (b) gracz losowy, (c) mieszany - minimax, który z prawdopodobieństwem ε wykonuje ruch losowy. Jeśli LLM i tak zawsze przegrywa, główną miarą staje się **długość partii** (jak długo model się bronił).
    - Możliwe dodatkowe wymiary (jeśli czas pozwoli): rozmiar planszy (8x8, 10x10, 15x15), głębokość/siła przeciwnika, model LLM.
    - Porównanie: odsetek wygranych i długość partii każdego wariantu względem punktu odniesienia, a także efekty główne obu czynników i ich interakcja dla każdej kombinacji (model, strona, przeciwnik); sprawdzamy, czy różnice są istotne statystycznie, i raportujemy przedziały ufności.
  - **procedura krok po kroku (przebieg eksperymentu):**
    1. Ustalić zestaw warunków: modele, strona, przeciwnik, rozmiar planszy, liczba partii na kombinację.
    2. Pilotaż (kilka partii na model): sprawdzić, czy przeciwnik ma odpowiednią siłę, ile trwa partia i czy mieścimy się w limitach API i kontekstu.
    3. Dla każdej kombinacji rozegrać N partii w każdym z 4 wariantów (w trybie rozmowy każda partia w osobnym czacie; ta sama pozycja startowa).
    4. Dla każdej partii zapisać wariant, wynik, długość (liczba tur), liczbę nielegalnych ruchów, liczbę tokenów i czas.
    5. Policzyć odsetek wygranych i średnią długość partii w każdym wariancie, wykonać porównania opisane wyżej.
    6. Porównać warianty między sobą oraz z punktem odniesienia, opisać wnioski.
  - **jakie narzędzia informatyczne, techniczne:** Python ≥ 3.13, `uv`, `openai` (API zgodne z OpenAI), `pydantic` (structured outputs), `rich` (UI); do analizy: `scipy`, `pandas`, `matplotlib`; backendy LLM zgodne z API OpenAI (chmurowe lub lokalna Ollama).
  - **punkt odniesienia:** analiza **random vs random** (jaki odsetek wygranych i jaka długość partii wynikają z samej struktury gry – dolna granica, którą LLM powinien przewyższać) oraz **minimax vs minimax** (jak wyglądają wyniki i długość partii przy rozsądnej grze obu stron – górna granica). Wyniki każdego wariantu LLM zestawiamy z tymi wartościami, aby ocenić, czy gra LLM jest lepsza od losowej i jak daleko jej do gry algorytmicznej. Dodatkowo wariant bez stanu z bezpośrednią odpowiedzią jest wewnętrznym punktem odniesienia dla pozostałych wariantów.

- **potrzebne zasoby:**
  - **jakie dane (jak pozyskane / wygenerowane):** brak danych zewnętrznych; dane powstają w symulacji - logi partii: model, strona, wariant, przeciwnik, wynik, liczba tur, nielegalne ruchy, tokeny, czas.
  - **co jeszcze jest potrzebne:** klucze API i limity (`config.json`), budżet na zapytania (rosnący kontekst zwiększa koszt), opcjonalnie GPU/Ollama dla modeli lokalnych.

- **zmienne zależne i niezależne:**
  - **co będzie zmieniane:**
    - niezależne główne: rozumowanie (bezpośrednia odpowiedź / CoT) oraz kontekst (bez stanu / rozmowa);
    - dodatkowe wymiary: model LLM, strona (wilk / owce), siła przeciwnika, rozmiar planszy.
  - **co będzie mierzone / obserwowane:**
    - odsetek wygranych LLM (główna miara);
    - **długość partii** – liczba tur, jak długo model się bronił;
    - pomocniczo: odsetek nielegalnych ruchów, liczba tokenów i czas na ruch.

- **oczekiwane wyniki:**
  - **jakie wyniki są możliwe i jak zaadresują hipotezy:**
    - CoT (lub rozmowa) daje istotnie wyższy odsetek wygranych lub dłuższe partie niż bezpośrednia odpowiedź (lub niezależne prompty) → potwierdzenie H1 (H2);
    - brak istotnej różnicy → brak podstaw do potwierdzenia;
    - wariant gorszy od swojego odpowiednika → odrzucenie (np. błędne rozumowanie małych modeli, gubienie się w długiej historii);
    - różny efekt CoT w zależności od trybu kontekstu → potwierdzenie H3;
    - efekt może zależeć od modelu i strony, raportowane osobno.
  - **co dodatkowego można sprawdzić / zrobić / uzyskać:** wpływ rozmiaru planszy; wpływ siły przeciwnika; porównanie ze strategią z _Winning Ways_ (owce teoretycznie zawsze wygrywają); koszt (tokeny) na przyrost skuteczności.

- **zadania do wykonania i ich alokacja:**
  1.  4 warianty w `LLMPlayer`: bezpośrednia odpowiedź / CoT (odpowiedź jako numer ruchu [1..n]; w CoT osobny prompt z analizą przed promptem wyboru ruchu) oraz bez stanu / rozmowa (jeden czat na partię, pełny stan w każdej wiadomości),
  2.  osłabiony / mieszany przeciwnik,
  3.  skrypt uruchamiający wiele partii i zapisujący logi (tokeny, czas, wynik, liczba tur),
  4.  pilotaż,
  5.  przebieg eksperymentu,
  6.  analiza statystyczna i wykresy,
  7.  opracowanie wyników.

- **co wymaga poznania / napisania:** dobór przeciwnika o odpowiedniej sile; dobór liczby partii; metody porównania wyników; limity i koszty API oraz okno kontekstu przy rosnącej historii.

- **braki, trudności, ryzyka:**
  - Nie wiemy jeszcze, jaki przeciwnik będzie się nadawał do porównania; przy dobrej grze owiec wilk jest teoretycznie skazany na porażkę – możliwy efekt podłogi.
  - Za mało partii → niska moc testu; nieznana wariancja wyników LLM.
  - Rosnący kontekst w trybie rozmowy: koszt, limity okna kontekstu, wpływ długiej historii na jakość odpowiedzi; CoT zwiększa to jeszcze bardziej.
  - Modele „myślące” rozumują same z siebie, co zaciera różnicę między bezpośrednią odpowiedzią a CoT; należy wyłączyć ten tryb lub wybierać modele bez niego.
  - 4 warianty zamiast 2 wymagają więcej partii dla tej samej mocy testu.
  - Limity darmowych API i różnice w obsłudze structured output między dostawcami.
  - Nieustalone: ostateczna liczba partii i wymiarów eksperymentu.
