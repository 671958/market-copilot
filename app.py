import re
from datetime import datetime, timezone
from urllib.parse import quote

import feedparser
import numpy as np
import pandas as pd
import requests
import streamlit as st
import yfinance as yf


st.set_page_config(
    page_title="Market Copilot",
    layout="centered",
)

st.title("Market Copilot")
st.caption("USA, Europa e Italia — ricerca personale sui mercati")
st.warning(
    "Dati gratuiti potenzialmente ritardati o incompleti. "
    "Punteggi descrittivi, non probabilità di guadagno. "
    "Nessuna esecuzione di ordini."
)

DEFAULT = (
    "AAPL, MSFT, NVDA, AMZN, GOOGL, "
    "ENEL.MI, ENI.MI, ISP.MI, UCG.MI, LDO.MI, "
    "ASML.AS, SAP.DE, AIR.PA"
)

BENCHMARKS = {
    "USA": {
        "S&P 500": "^GSPC",
        "Nasdaq 100": "^NDX",
    },
    "Europa e Italia": {
        "Euro Stoxx 50": "^STOXX50E",
        "DAX": "^GDAXI",
        "FTSE MIB": "FTSEMIB.MI",
    },
}


def finite(value):
    try:
        number = float(value)
        return number if np.isfinite(number) else None
    except (ValueError, TypeError):
        return None


def pct(value):
    number = finite(value)
    return None if number is None else round(number * 100, 2)


@st.cache_data(ttl=1800, show_spinner=False)
def history(symbol):
    frame = yf.Ticker(symbol).history(
        period="2y",
        interval="1d",
        auto_adjust=True,
    )
    if frame.empty or "Close" not in frame:
        raise ValueError("Storico non disponibile")
    frame = frame.sort_index()
    frame = frame.loc[~frame.index.duplicated(keep="last")]
    frame = frame.dropna(subset=["Close"])
    frame = frame[frame["Close"] > 0]
    if len(frame) < 201:
        raise ValueError(
            "Servono almeno 201 sedute: storico insufficiente "
            "per questa metodologia"
        )
    return frame


@st.cache_data(ttl=21600, show_spinner=False)
def fundamentals(symbol):
    return yf.Ticker(symbol).info


def calculate(frame):
    close = frame["Close"]
    last = float(close.iloc[-1])
    ma50 = float(close.rolling(50).mean().iloc[-1])
    ma200 = float(close.rolling(200).mean().iloc[-1])
    r1 = (last / float(close.iloc[-22]) - 1) * 100
    r3 = (last / float(close.iloc[-64]) - 1) * 100

    changes = close.diff()
    gain = changes.clip(lower=0).rolling(14).mean().iloc[-1]
    loss = (-changes.clip(upper=0)).rolling(14).mean().iloc[-1]
    if gain == 0 and loss == 0:
        rsi = 50.0
    elif loss == 0:
        rsi = 100.0
    else:
        rsi = 100 - 100 / (1 + gain / loss)

    conditions = [
        last < ma50,
        last < ma200,
        ma50 < ma200,
        r1 < 0,
        r3 < 0,
    ]
    weakness = sum(int(x) for x in conditions) * 20

    volume_ratio = None
    if "Volume" in frame:
        previous = finite(frame["Volume"].iloc[-21:-1].mean())
        current = finite(frame["Volume"].iloc[-1])
        if previous is not None and previous > 0 and current is not None:
            volume_ratio = round(current / previous, 2)

    return {
        "Ultima seduta": str(frame.index[-1].date()),
        "Prezzo rettificato": round(last, 4),
        "Rendimento 21 sedute %": round(r1, 2),
        "Rendimento 63 sedute %": round(r3, 2),
        "MA50": round(ma50, 4),
        "MA200": round(ma200, 4),
        "RSI semplice 14": round(float(rsi), 1),
        "Debolezza tecnica /100": weakness,
        "Volume / media precedente 20": volume_ratio,
    }


def parse_symbols(text):
    candidates = re.split(r"[\s,;]+", text.upper().strip())
    return list(dict.fromkeys(x for x in candidates if x))


def classify(score):
    if score >= 60:
        return "Debolezza tecnica elevata"
    if score >= 40:
        return "Quadro tecnico misto"
    return "Debolezza tecnica contenuta"


@st.cache_data(ttl=1800, show_spinner=False)
def news(query):
    url = (
        "https://news.google.com/rss/search?q="
        + quote(query)
        + "&hl=it&gl=IT&ceid=IT:it"
    )
    response = requests.get(
        url,
        timeout=20,
        headers={"User-Agent": "MarketCopilotPersonal/1.0"},
    )
    response.raise_for_status()
    parsed = feedparser.parse(response.content)
    items = []
    for entry in parsed.entries[:15]:
        link = entry.get("link", "")
        if not link.startswith(("https://", "http://")):
            continue
        source = entry.get("source", {})
        items.append({
            "Titolo": entry.get("title", "Senza titolo"),
            "Pubblicazione": entry.get("published", "Data non disponibile"),
            "Fonte": source.get("title", "Non indicata"),
            "Link": link,
        })
    return items


page = st.selectbox(
    "Scegli una sezione",
    ["Mercato", "Analizza un titolo", "Screening", "Notizie", "Metodo e limiti"],
)

if st.button("Svuota cache e aggiorna"):
    st.cache_data.clear()
    st.rerun()


if page == "Mercato":
    st.subheader("Condizioni tecniche degli indici")
    st.caption(
        "USA ed Europa sono valutati separatamente. "
        "Non è una previsione di ribasso."
    )
    for region, indexes in BENCHMARKS.items():
        st.markdown("### " + region)
        rows = []
        for name, symbol in indexes.items():
            try:
                result = calculate(history(symbol))
                rows.append({"Indice": name, **result})
            except Exception as exc:
                st.warning(f"{name}: dati non disponibili ({exc})")
        if rows:
            table = pd.DataFrame(rows)
            score = table["Debolezza tecnica /100"].mean()
            st.metric("Debolezza media degli indici disponibili", f"{score:.0f}/100")
            st.write(classify(score))
            st.dataframe(table, hide_index=True, use_container_width=True)
            st.caption(
                f"Copertura: {len(rows)}/{len(indexes)} indici. "
                "Le medie possono cambiare se mancano dati."
            )
        else:
            st.error("Dati insufficienti: nessuna valutazione disponibile.")


elif page == "Analizza un titolo":
    symbol = st.text_input("Simbolo del titolo", "ENEL.MI").strip().upper()
    if st.button("Analizza", type="primary") and symbol:
        try:
            with st.spinner("Recupero dati..."):
                frame = history(symbol)
                result = calculate(frame)

            st.subheader(symbol)
            st.write(classify(result["Debolezza tecnica /100"]))
            st.dataframe(
                pd.DataFrame(result.items(), columns=["Indicatore", "Valore"]),
                hide_index=True,
                use_container_width=True,
            )
            chart = pd.DataFrame({
                "Prezzo rettificato": frame["Close"],
                "Media 50": frame["Close"].rolling(50).mean(),
                "Media 200": frame["Close"].rolling(200).mean(),
            })
            st.line_chart(chart.tail(252))

            try:
                info = fundamentals(symbol)
                st.write("Società:", info.get("longName", symbol))
                st.write("Valuta quotazione:", info.get("currency", "Non disponibile"))
                st.write(
                    "Valuta bilancio:",
                    info.get("financialCurrency", "Non disponibile"),
                )
                details = {
                    "Crescita ricavi % (campo del fornitore)": pct(info.get("revenueGrowth")),
                    "Margine operativo %": pct(info.get("operatingMargins")),
                    "Debt/equity (campo del fornitore)": finite(info.get("debtToEquity")),
                    "P/E storico": finite(info.get("trailingPE")),
                    "Capitalizzazione in valuta quotazione": finite(info.get("marketCap")),
                    "Flusso cassa operativo in valuta bilancio": finite(info.get("operatingCashflow")),
                }
                st.dataframe(
                    pd.DataFrame(details.items(), columns=["Dato", "Valore"]),
                    hide_index=True,
                    use_container_width=True,
                )
                st.caption(
                    "I dati fondamentali possono riferirsi a periodi diversi. "
                    "Verificali nei bilanci ufficiali. Debt/equity non è debito/EBITDA."
                )
            except Exception as exc:
                st.warning(f"Fondamentali non disponibili: {exc}")

        except Exception as exc:
            st.error(f"Analisi non disponibile: {exc}")


elif page == "Screening":
    st.subheader("Confronta una lista di aziende")
    st.caption(
        "Ricerca solo nei simboli inseriti, non in tutte le borse. "
        "La lista iniziale è dimostrativa, non una selezione di aziende emergenti."
    )
    text = st.text_area("Simboli separati da virgole", DEFAULT, height=150)
    include_fundamentals = st.checkbox(
        "Aggiungi fondamentali: più lento, possibili limiti del fornitore",
        value=False,
    )
    if st.button("Avvia confronto", type="primary"):
        symbols = parse_symbols(text)
        if not symbols:
            st.warning("Inserisci almeno un simbolo.")
        elif len(symbols) > 30:
            st.warning("Massimo 30 simboli per questa versione.")
        else:
            rows = []
            errors = []
            progress = st.progress(0)
            for index, symbol in enumerate(symbols):
                try:
                    result = calculate(history(symbol))
                    row = {"Simbolo": symbol, **result}
                    if include_fundamentals:
                        try:
                            info = fundamentals(symbol)
                            row.update({
                                "Nome": info.get("shortName", symbol),
                                "Valuta": info.get("currency"),
                                "Crescita ricavi %": pct(info.get("revenueGrowth")),
                                "Margine operativo %": pct(info.get("operatingMargins")),
                                "Debt/equity": finite(info.get("debtToEquity")),
                            })
                        except Exception as exc:
                            errors.append(f"{symbol}: fondamentali mancanti ({exc})")
                    rows.append(row)
                except Exception as exc:
                    errors.append(f"{symbol}: {exc}")
                progress.progress((index + 1) / len(symbols))

            if rows:
                table = pd.DataFrame(rows).sort_values(
                    "Rendimento 63 sedute %",
                    ascending=False,
                )
                st.dataframe(table, hide_index=True, use_container_width=True)
                st.download_button(
                    "Esporta risultati CSV",
                    table.to_csv(index=False).encode("utf-8-sig"),
                    "screening.csv",
                    "text/csv",
                )
                st.caption(
                    "Ordinamento per rendimento passato a 63 sedute: "
                    "non indica crescita futura. La debolezza tecnica "
                    "non verifica disponibilità o costo dello short."
                )
            if errors:
                with st.expander("Errori e dati mancanti"):
                    for error in errors:
                        st.write(error)


elif page == "Notizie":
    st.subheader("Ricerca di notizie pubbliche")
    query = st.text_input(
        "Azienda o tema da cercare",
        "BCE OR Federal Reserve OR inflazione",
    )
    st.caption(
        "Ricerca di titoli e collegamenti: non lettura integrale degli articoli, "
        "non verifica indipendente e non rilevamento di informazioni riservate."
    )
    if st.button("Cerca notizie") and query.strip():
        try:
            items = news(query.strip())
            if not items:
                st.info("Nessuna notizia restituita dalla fonte.")
            for item in items:
                st.write(item["Titolo"])
                st.caption(item["Fonte"] + " | " + item["Pubblicazione"])
                st.link_button("Apri la fonte", item["Link"])
                st.divider()
        except Exception as exc:
            st.error(f"Notizie non disponibili: {exc}")


else:
    st.subheader("Metodo e limiti")
    st.write(
        "La debolezza tecnica assegna 20 punti a ciascuna condizione: "
        "prezzo sotto MA50; prezzo sotto MA200; MA50 sotto MA200; "
        "rendimento a 21 sedute negativo; rendimento a 63 sedute negativo."
    )
    st.write(
        "Il punteggio va da 0 a 100. Non è una probabilità e non è "
        "stato validato come strategia d'investimento. "
        "Gli indicatori sono correlati e possono produrre falsi segnali."
    )
    st.write(
        "RSI: versione con medie semplici, non smoothing di Wilder. "
        "Prezzi rettificati dal fornitore: possono differire dai prezzi "
        "effettivamente scambiati. L'ultima barra giornaliera può essere "
        "ancora incompleta a mercato aperto."
    )
    st.write(
        "Aggiornamento al caricamento, con cache: non streaming continuo. "
        "Nessuna notifica automatica, archivio persistente, "
        "backtest o connessione a broker in questa versione."
    )
    st.write(
        "Lo short diretto può comportare perdite teoricamente illimitate, "
        "costi di prestito e richieste di margine. "
        "Questa app non verifica tali condizioni."
    )
    st.write(
        "Non inserire password, token, saldo o dati del conto. "
        "Se pubblichi l'app senza restrizioni di accesso, "
        "non considerarla uno spazio privato."
    )

st.caption(
    "Pagina eseguita: "
    + datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    + " — non è l'orario di aggiornamento delle quotazioni."
)
st.divider()

with st.expander("Spiegami i segnali di un titolo", expanded=False):
    st.write(
        "Questo pannello spiega le regole tecniche del programma. "
        "Non formula raccomandazioni di acquisto, vendita o short."
    )

    explanation_symbol = st.text_input(
        "Simbolo da spiegare",
        value="ENEL.MI",
        key="explanation_symbol",
    ).strip().upper()

    if st.button("Genera spiegazione", key="explanation_button"):
        if not explanation_symbol:
            st.warning("Inserisci un simbolo.")
        else:
            try:
                with st.spinner("Preparazione della spiegazione..."):
                    explanation_data = history(explanation_symbol)
                    explanation_result = calculate(explanation_data)

                price = explanation_result["Prezzo rettificato"]
                average50 = explanation_result["MA50"]
                average200 = explanation_result["MA200"]
                return21 = explanation_result["Rendimento 21 sedute %"]
                return63 = explanation_result["Rendimento 63 sedute %"]
                weakness = explanation_result["Debolezza tecnica /100"]
                rsi_value = explanation_result["RSI semplice 14"]
                volume_value = explanation_result[
                    "Volume / media precedente 20"
                ]

                st.subheader(explanation_symbol)
                st.write(
                    "Seduta dei dati:",
                    explanation_result["Ultima seduta"],
                )
                st.metric("Debolezza tecnica", f"{weakness}/100")
                st.write(classify(weakness))

                explanation_close = explanation_data["Close"]
                exact_price = float(explanation_close.iloc[-1])
                exact_ma50 = float(
                    explanation_close.rolling(50).mean().iloc[-1]
                )
                exact_ma200 = float(
                    explanation_close.rolling(200).mean().iloc[-1]
                )
                exact_return21 = (
                    exact_price / float(explanation_close.iloc[-22]) - 1
                ) * 100
                exact_return63 = (
                    exact_price / float(explanation_close.iloc[-64]) - 1
                ) * 100

                checks = [
                    ("Prezzo sotto MA50", exact_price < exact_ma50),
                    ("Prezzo sotto MA200", exact_price < exact_ma200),
                    ("MA50 sotto MA200", exact_ma50 < exact_ma200),
                    ("Rendimento 21 sedute negativo", exact_return21 < 0),
                    ("Rendimento 63 sedute negativo", exact_return63 < 0),
                ]

                st.write("Da dove viene il punteggio:")
                for label, active in checks:
                    points = 20 if active else 0
                    answer = "Sì" if active else "No"
                    st.write(f"- {label}: {answer}. Punti: {points}.")

                st.write("Lettura dei periodi:")
                st.write(
                    f"- Nelle ultime 21 sedute il rendimento "
                    f"rettificato è {return21:+.2f}%."
                )
                st.write(
                    f"- Nelle ultime 63 sedute il rendimento "
                    f"rettificato è {return63:+.2f}%."
                )

                if exact_return21 > 0 and exact_return63 < 0:
                    st.info(
                        "Il periodo più recente è positivo, ma quello "
                        "più lungo è negativo. È una divergenza tra "
                        "orizzonti temporali, non una conferma "
                        "automatica di inversione."
                    )
                elif exact_return21 < 0 and exact_return63 > 0:
                    st.info(
                        "Il periodo più recente è negativo, mentre "
                        "quello più lungo resta positivo. I due "
                        "orizzonti mostrano direzioni diverse."
                    )
                else:
                    st.info(
                        "Consulta insieme rendimenti e medie mobili: "
                        "nessuna singola condizione basta per "
                        "prevedere il prossimo movimento."
                    )

                st.write(
                    f"RSI semplice 14: {rsi_value}. "
                    "È mostrato come informazione aggiuntiva, "
                    "ma non entra nel punteggio di debolezza."
                )

                if volume_value is not None:
                    st.write(
                        f"Volume dell'ultima seduta: "
                        f"{volume_value:.2f} volte la media "
                        "delle 20 sedute precedenti."
                    )
                else:
                    st.write("Confronto dei volumi non disponibile.")

                st.caption(
                    "Prezzi e medie sono espressi nella valuta "
                    "del titolo. I confronti usano valori non "
                    "arrotondati. L'ultima barra può essere "
                    "incompleta se il mercato è aperto."
                )

                st.warning(
                    "Prima di una decisione restano da verificare "
                    "bilanci, valutazione, notizie ufficiali, "
                    "liquidità e rischi. Questo pannello non "
                    "conferma la convenienza di un investimento."
                )

            except Exception as exc:
                st.error(f"Spiegazione non disponibile: {exc}")
                st.divider()
st.header("Ricerca aziende e monitor notizie")

RESEARCH_LISTS = {
    "Italia - campione iniziale": (
        "ENEL.MI, ENI.MI, ISP.MI, UCG.MI, LDO.MI, "
        "MONC.MI, PRY.MI, STM.MI, DIA.MI, AMPL.MI"
    ),
    "Europa - campione iniziale": (
        "ASML.AS, SAP.DE, AIR.PA, SIE.DE, "
        "DSY.PA, ASM.AS, BESI.AS, ADYEN.AS"
    ),
    "USA - campione iniziale": (
        "MSFT, NVDA, AMD, PLTR, CRWD, NET, "
        "DDOG, SNOW, SOFI, RKLB"
    ),
    "Lista personale": "",
}

st.caption(
    "Campioni dimostrativi, non elenchi completi o raccomandazioni. "
    "Non tutte queste aziende sono piccole, nuove o emergenti. "
    "I simboli e i dati vengono verificati al momento della richiesta."
)

research_group = st.selectbox(
    "Scegli la lista da usare",
    list(RESEARCH_LISTS),
    key="research_group_v2",
)

research_text = st.text_area(
    "Modifica o aggiungi simboli separati da virgole",
    value=RESEARCH_LISTS[research_group],
    height=130,
    key="research_symbols_" + research_group,
)

st.subheader("Filtri fondamentali")

minimum_growth = st.number_input(
    "Crescita minima ricavi, in percentuale",
    min_value=-100.0,
    max_value=1000.0,
    value=10.0,
    step=5.0,
    key="research_min_growth",
)

positive_margin = st.checkbox(
    "Richiedi margine operativo positivo",
    value=True,
    key="research_positive_margin",
)

st.caption(
    "Un'azienda senza i dati richiesti viene indicata come incompleta, "
    "non trattata come se avesse valori pari a zero. "
    "Le soglie sono filtri di ricerca, non una strategia validata. "
    "Non viene applicato un filtro di capitalizzazione perché le "
    "valute di quotazione possono essere diverse."
)

if st.button("Cerca candidate nella lista", key="research_start_v2"):
    research_symbols = parse_symbols(research_text)

    if not research_symbols:
        st.warning("Inserisci almeno un simbolo.")
    elif len(research_symbols) > 20:
        st.warning(
            "Usa al massimo 20 simboli per ricerca per limitare "
            "carico e richieste alla fonte gratuita."
        )
    else:
        research_rows = []
        research_errors = []
        research_progress = st.progress(0)

        for position, research_symbol in enumerate(research_symbols):
            try:
                research_info = fundamentals(research_symbol)
                if not isinstance(research_info, dict) or not research_info:
                    raise ValueError("Scheda fondamentale vuota")

                growth_value = pct(research_info.get("revenueGrowth"))
                margin_value = pct(research_info.get("operatingMargins"))

                required_missing = []
                if growth_value is None:
                    required_missing.append("crescita ricavi")
                if positive_margin and margin_value is None:
                    required_missing.append("margine operativo")

                if required_missing:
                    filter_result = (
                        "Dati incompleti: " + ", ".join(required_missing)
                    )
                elif growth_value < minimum_growth:
                    filter_result = "Non supera filtro ricavi"
                elif positive_margin and margin_value <= 0:
                    filter_result = "Non supera filtro margine"
                else:
                    filter_result = "Supera i filtri"

                research_row = {
                    "Simbolo": research_symbol,
                    "Nome": research_info.get("shortName", research_symbol),
                    "Esito": filter_result,
                    "Settore": research_info.get("sector", "Non disponibile"),
                    "Paese società": research_info.get(
                        "country", "Non disponibile"
                    ),
                    "Valuta quotazione": research_info.get("currency"),
                    "Valuta bilancio": research_info.get("financialCurrency"),
                    "Crescita ricavi %": growth_value,
                    "Margine operativo %": margin_value,
                    "P/E storico": finite(
                        research_info.get("trailingPE")
                    ),
                    "Debt/equity del fornitore": finite(
                        research_info.get("debtToEquity")
                    ),
                    "Capitalizzazione in valuta quotazione": finite(
                        research_info.get("marketCap")
                    ),
                    "Flusso operativo in valuta bilancio": finite(
                        research_info.get("operatingCashflow")
                    ),
                }

                fiscal_timestamp = finite(
                    research_info.get("mostRecentQuarter")
                )
                if fiscal_timestamp is not None:
                    try:
                        fiscal_label = datetime.fromtimestamp(
                            fiscal_timestamp, tz=timezone.utc
                        ).date().isoformat()
                    except (ValueError, OverflowError, OSError):
                        fiscal_label = "Non disponibile"
                else:
                    fiscal_label = "Non disponibile"

                research_row[
                    "Trimestre recente indicato dal fornitore"
                ] = fiscal_label

                try:
                    research_technical = calculate(
                        history(research_symbol)
                    )
                    research_row.update({
                        "Ultima seduta": research_technical["Ultima seduta"],
                        "Rendimento 63 sedute %": research_technical[
                            "Rendimento 63 sedute %"
                        ],
                        "Debolezza tecnica /100": research_technical[
                            "Debolezza tecnica /100"
                        ],
                        "Copertura tecnica": "Disponibile",
                    })
                except Exception as technical_error:
                    research_row["Copertura tecnica"] = (
                        "Non disponibile: " + str(technical_error)
                    )

                research_rows.append(research_row)

            except Exception as research_error:
                research_errors.append(
                    f"{research_symbol}: {research_error}"
                )

            research_progress.progress(
                (position + 1) / len(research_symbols)
            )

        st.session_state["research_result_v2"] = {
            "rows": research_rows,
            "errors": research_errors,
            "requested": len(research_symbols),
            "growth_threshold": minimum_growth,
            "positive_margin": positive_margin,
            "completed": datetime.now(timezone.utc).strftime(
                "%Y-%m-%d %H:%M UTC"
            ),
        }

if "research_result_v2" in st.session_state:
    saved_research = st.session_state["research_result_v2"]

    st.caption(
        "Ultima ricerca completata: "
        + saved_research["completed"]
        + ". È l'orario del calcolo, non di pubblicazione dei bilanci. "
        "I risultati restano quelli dell'ultima ricerca avviata."
    )

    st.write(
        "Filtri usati:",
        f"crescita ricavi almeno {saved_research['growth_threshold']}%; "
        f"margine positivo richiesto: "
        f"{'sì' if saved_research['positive_margin'] else 'no'}.",
    )

    if saved_research["rows"]:
        research_table = pd.DataFrame(saved_research["rows"])
        candidate_table = research_table[
            research_table["Esito"] == "Supera i filtri"
        ].copy()

        st.write(
            f"Schede recuperate: {len(research_table)}/"
            f"{saved_research['requested']}. "
            f"Società che superano i filtri: {len(candidate_table)}."
        )

        if not candidate_table.empty:
            st.subheader("Candidate da approfondire")
            st.dataframe(
                candidate_table,
                hide_index=True,
                use_container_width=True,
            )
        else:
            st.info(
                "Nessuna candidata con i filtri usati e i dati disponibili."
            )

        with st.expander("Tutti i risultati e i dati mancanti"):
            st.dataframe(
                research_table,
                hide_index=True,
                use_container_width=True,
            )

        st.download_button(
            "Esporta ricerca aziende",
            research_table.to_csv(index=False).encode("utf-8-sig"),
            file_name="ricerca_aziende.csv",
            mime="text/csv",
            key="research_export_v2",
        )

        st.caption(
            "Il trimestre indicato non certifica il periodo di ogni campo: "
            "ricavi, margini e multipli possono riferirsi a basi diverse. "
            "Verifica bilanci ufficiali, diluizione delle azioni, "
            "debito, concorrenza e valutazione. Superare questi filtri "
            "non dimostra che un'azienda sia emergente o sottovalutata."
        )

    if saved_research["errors"]:
        with st.expander("Società non analizzate ed errori"):
            for research_message in saved_research["errors"]:
                st.write(research_message)


@st.cache_data(ttl=60, show_spinner=False)
def research_news_feed(query_text):
    feed_url = (
        "https://news.google.com/rss/search?q="
        + quote(query_text)
        + "&hl=it&gl=IT&ceid=IT:it"
    )
    feed_response = requests.get(
        feed_url,
        timeout=15,
        headers={"User-Agent": "MarketCopilotPersonal/1.0"},
    )
    feed_response.raise_for_status()
    feed_data = feedparser.parse(feed_response.content)

    feed_rows = []
    seen_links = set()

    for entry in feed_data.entries:
        entry_link = entry.get("link", "")
        if not entry_link.startswith(("https://", "http://")):
            continue
        if entry_link in seen_links:
            continue

        seen_links.add(entry_link)
        feed_source = entry.get("source", {})
        feed_rows.append({
            "title": entry.get("title", "Senza titolo"),
            "source": feed_source.get("title", "Non indicata"),
            "published": entry.get("published", "Data non disponibile"),
            "link": entry_link,
        })
        if len(feed_rows) >= 12:
            break

    return {
        "items": feed_rows,
        "retrieved": datetime.now(timezone.utc).strftime(
            "%Y-%m-%d %H:%M:%S UTC"
        ),
    }


st.divider()
st.subheader("Monitor notizie pubbliche")

st.write(
    "Inserisci nomi di aziende o temi. Il monitor cerca titoli e "
    "collegamenti, non legge integralmente gli articoli e non "
    "verifica automaticamente le affermazioni."
)

monitor_query = st.text_input(
    "Ricerca da monitorare",
    value='("IPO" OR "quotazione" OR "nuovo finanziamento") when:7d',
    key="research_monitor_query_v2",
)

monitor_enabled = st.checkbox(
    "Attiva il controllo periodico delle notizie",
    value=False,
    key="research_monitor_enabled_v2",
)

monitor_seconds = st.selectbox(
    "Intervallo di controllo, in secondi",
    options=[60, 300, 900],
    index=1,
    key="research_monitor_seconds_v2",
)

st.caption(
    "Consigliato: 300 secondi. Il controllo a 60 secondi non "
    "garantisce notizie nuove ogni minuto. Funziona mentre la "
    "sessione è attiva; non è un servizio di avvisi in background. "
    "La ricerca può includere aziende non quotate, senza prezzi "
    "o fondamentali disponibili nell'app."
)


@st.fragment(
    run_every=monitor_seconds if monitor_enabled else None
)
def render_research_monitor():
    if not monitor_enabled:
        st.info("Monitor fermo. Attiva la casella per avviarlo.")
        return

    if not monitor_query.strip():
        st.warning("Inserisci una ricerca.")
        return

    try:
        result_feed = research_news_feed(monitor_query.strip())

        st.caption(
            "Ultima risposta recuperata dalla fonte: "
            + result_feed["retrieved"]
            + ". Non è l'orario di pubblicazione delle notizie."
        )

        if not result_feed["items"]:
            st.info("Nessun risultato restituito dalla fonte.")

        for monitor_item in result_feed["items"]:
            st.write(monitor_item["title"])
            st.caption(
                monitor_item["source"]
                + " | "
                + monitor_item["published"]
            )
            st.link_button(
                "Apri articolo",
                monitor_item["link"],
            )
            st.divider()

    except Exception as monitor_error:
        st.error(f"Monitor temporaneamente non disponibile: {monitor_error}")


render_research_monitor()
st.divider()
st.header("Dimensione e profilo di crescita")

st.caption(
    "Filtri sulle schede dell'ultima ricerca aziende. "
    "Non esplora nuove società e non modifica i risultati precedenti. "
    "Le soglie sono scelte di ricerca, non classificazioni ufficiali "
    "né raccomandazioni di investimento."
)

if "research_result_v2" not in st.session_state:
    st.info(
        "Prima esegui 'Cerca candidate nella lista' "
        "nel pannello Ricerca aziende."
    )
else:
    profile_source = st.session_state["research_result_v2"]
    profile_rows = profile_source.get("rows", [])

    if not profile_rows:
        st.warning("L'ultima ricerca non contiene schede utilizzabili.")
    else:
        profile_df = pd.DataFrame(profile_rows).copy()

        profile_currency_col = "Valuta quotazione"
        profile_cap_col = "Capitalizzazione in valuta quotazione"
        profile_growth_col = "Crescita ricavi %"
        profile_margin_col = "Margine operativo %"

        for profile_column in [
            profile_currency_col,
            profile_cap_col,
            profile_growth_col,
            profile_margin_col,
        ]:
            if profile_column not in profile_df:
                profile_df[profile_column] = None

        profile_currencies = sorted({
            str(value).strip()
            for value in profile_df[profile_currency_col].dropna()
            if str(value).strip()
        })

        if not profile_currencies:
            st.warning(
                "Valute di quotazione mancanti: "
                "non posso applicare il filtro dimensionale."
            )
        else:
            profile_currency = st.selectbox(
                "Valuta di quotazione da analizzare",
                options=profile_currencies,
                key="profile_currency_v3",
            )

            st.write(
                "Gli importi qui sotto sono in milioni di",
                profile_currency,
                "e sono confrontati con la capitalizzazione "
                "riportata dal fornitore nella stessa valuta.",
            )

            profile_min_cap = st.number_input(
                "Capitalizzazione minima, in milioni",
                min_value=0.0,
                value=100.0,
                step=100.0,
                key="profile_min_cap_v3",
            )

            profile_max_cap = st.number_input(
                "Capitalizzazione massima, in milioni",
                min_value=0.0,
                value=10000.0,
                step=500.0,
                key="profile_max_cap_v3",
            )

            profile_min_growth = st.number_input(
                "Crescita minima ricavi %",
                min_value=-100.0,
                max_value=1000.0,
                value=10.0,
                step=5.0,
                key="profile_min_growth_v3",
            )

            profile_margin_mode = st.selectbox(
                "Profilo operativo",
                options=[
                    "Tutti i margini disponibili",
                    "Solo margine operativo positivo",
                    "Solo margine operativo negativo",
                ],
                key="profile_margin_mode_v3",
            )

            st.caption(
                "Esempio di fascia impostata: da 100 milioni "
                "a 10 miliardi nella valuta selezionata. "
                "Puoi cambiarla. Una capitalizzazione contenuta "
                "non dimostra che un'azienda sia giovane o conveniente."
            )

            if profile_max_cap < profile_min_cap:
                st.error(
                    "La capitalizzazione massima deve essere "
                    "almeno uguale alla minima."
                )
            else:
                profile_same_currency = profile_df[
                    profile_df[profile_currency_col] == profile_currency
                ].copy()

                for profile_column in [
                    profile_cap_col,
                    profile_growth_col,
                    profile_margin_col,
                ]:
                    profile_same_currency[profile_column] = (
                        pd.to_numeric(
                            profile_same_currency[profile_column],
                            errors="coerce",
                        ).replace([np.inf, -np.inf], np.nan)
                    )

                profile_complete_mask = (
                    profile_same_currency[
                        [
                            profile_cap_col,
                            profile_growth_col,
                            profile_margin_col,
                        ]
                    ].notna().all(axis=1)
                    & (profile_same_currency[profile_cap_col] > 0)
                )

                profile_complete = profile_same_currency[
                    profile_complete_mask
                ].copy()

                profile_incomplete = profile_same_currency[
                    ~profile_complete_mask
                ].copy()

                profile_match_mask = (
                    profile_complete[profile_cap_col].between(
                        profile_min_cap * 1_000_000,
                        profile_max_cap * 1_000_000,
                        inclusive="both",
                    )
                    & (
                        profile_complete[profile_growth_col]
                        >= profile_min_growth
                    )
                )

                if profile_margin_mode == "Solo margine operativo positivo":
                    profile_match_mask &= (
                        profile_complete[profile_margin_col] > 0
                    )
                elif profile_margin_mode == "Solo margine operativo negativo":
                    profile_match_mask &= (
                        profile_complete[profile_margin_col] < 0
                    )

                profile_matches = profile_complete[
                    profile_match_mask
                ].copy()

                profile_matches[
                    "Capitalizzazione in milioni"
                ] = (
                    profile_matches[profile_cap_col] / 1_000_000
                ).round(2)

                profile_matches["Profilo descrittivo"] = (
                    profile_matches[profile_margin_col].apply(
                        lambda margin: (
                            "Margine operativo positivo"
                            if margin > 0
                            else (
                                "Margine operativo negativo"
                                if margin < 0
                                else "Margine operativo nullo"
                            )
                        )
                    )
                )

                profile_matches = profile_matches.sort_values(
                    profile_growth_col,
                    ascending=False,
                )

                st.caption(
                    "Schede dell'ultima ricerca completata: "
                    + profile_source.get("completed", "Orario non disponibile")
                    + ". I filtri non aggiornano i dati alla fonte."
                )

                st.write(
                    f"Schede totali: {len(profile_df)}. "
                    f"Nella valuta selezionata: "
                    f"{len(profile_same_currency)}. "
                    f"Complete per questi filtri: "
                    f"{len(profile_complete)}. "
                    f"Corrispondenze: {len(profile_matches)}."
                )

                if profile_matches.empty:
                    st.info(
                        "Nessuna corrispondenza. Non è un errore: "
                        "la lista può non contenere società "
                        "con questo profilo."
                    )
                else:
                    profile_display_columns = [
                        column
                        for column in [
                            "Simbolo",
                            "Nome",
                            "Settore",
                            profile_currency_col,
                            "Capitalizzazione in milioni",
                            profile_growth_col,
                            profile_margin_col,
                            "Profilo descrittivo",
                            "Ultima seduta",
                            "Copertura tecnica",
                        ]
                        if column in profile_matches.columns
                    ]

                    st.dataframe(
                        profile_matches[profile_display_columns],
                        hide_index=True,
                        use_container_width=True,
                    )

                    st.download_button(
                        "Esporta selezione per profilo",
                        profile_matches[
                            profile_display_columns
                        ].to_csv(index=False).encode("utf-8-sig"),
                        file_name="selezione_profilo.csv",
                        mime="text/csv",
                        key="profile_export_v3",
                    )

                if not profile_incomplete.empty:
                    with st.expander(
                        "Schede escluse per dati mancanti "
                        "o capitalizzazione non valida"
                    ):
                        st.dataframe(
                            profile_incomplete,
                            hide_index=True,
                            use_container_width=True,
                        )

                st.warning(
                    "Margine operativo positivo non significa "
                    "necessariamente utile netto o cassa positivi. "
                    "Crescita dei ricavi e dimensione non bastano: "
                    "restano da verificare bilanci, periodo dei dati, "
                    "diluizione, debito, liquidità e valutazione. "
                    "Il modulo non verifica lo stato di una IPO "
                    "né classifica aziende private."
                )
                st.divider()
st.header("Centro di controllo della selezione")

st.caption(
    "Diagnostica dell'ultima ricerca aziende. "
    "I colori indicano lo stato dei filtri, non la convenienza "
    "di un investimento."
)

dashboard_saved = st.session_state.get("research_result_v2")

if not dashboard_saved or not dashboard_saved.get("rows"):
    st.info(
        "Esegui prima una ricerca nel pannello "
        "'Ricerca aziende e monitor notizie'."
    )
else:
    dashboard_currency = st.session_state.get("profile_currency_v3")
    dashboard_min_cap = float(
        st.session_state.get("profile_min_cap_v3", 100.0)
    )
    dashboard_max_cap = float(
        st.session_state.get("profile_max_cap_v3", 10000.0)
    )
    dashboard_growth = float(
        st.session_state.get("profile_min_growth_v3", 10.0)
    )
    dashboard_margin_mode = st.session_state.get(
        "profile_margin_mode_v3",
        "Tutti i margini disponibili",
    )

    if not dashboard_currency:
        st.info(
            "Scegli prima una valuta nel pannello "
            "'Dimensione e profilo di crescita'."
        )
    elif dashboard_max_cap < dashboard_min_cap:
        st.error(
            "Correggi le soglie dimensionali: "
            "il massimo deve essere almeno uguale al minimo."
        )
    else:
        dashboard_rows = []

        for source_row in dashboard_saved["rows"]:
            symbol_value = source_row.get("Simbolo", "Non disponibile")
            currency_value = source_row.get("Valuta quotazione")
            cap_value = finite(
                source_row.get("Capitalizzazione in valuta quotazione")
            )
            growth_value = finite(
                source_row.get("Crescita ricavi %")
            )
            margin_value = finite(
                source_row.get("Margine operativo %")
            )

            missing_reasons = []
            failed_reasons = []

            if not currency_value:
                missing_reasons.append("valuta mancante")
            elif currency_value != dashboard_currency:
                failed_reasons.append("valuta diversa")

            if cap_value is None or cap_value <= 0:
                missing_reasons.append(
                    "capitalizzazione mancante o non valida"
                )
            else:
                if cap_value < dashboard_min_cap * 1_000_000:
                    failed_reasons.append("sotto dimensione minima")
                if cap_value > dashboard_max_cap * 1_000_000:
                    failed_reasons.append("sopra dimensione massima")

            if growth_value is None:
                missing_reasons.append("crescita ricavi mancante")
            elif growth_value < dashboard_growth:
                failed_reasons.append("crescita sotto soglia")

            if margin_value is None:
                missing_reasons.append("margine operativo mancante")
            elif (
                dashboard_margin_mode
                == "Solo margine operativo positivo"
                and margin_value <= 0
            ):
                failed_reasons.append("margine non positivo")
            elif (
                dashboard_margin_mode
                == "Solo margine operativo negativo"
                and margin_value >= 0
            ):
                failed_reasons.append("margine non negativo")

            if missing_reasons:
                status_value = "Dati insufficienti"
            elif failed_reasons:
                status_value = "Non supera i filtri"
            else:
                status_value = "Supera i filtri"

            all_reasons = missing_reasons + failed_reasons

            dashboard_rows.append({
                "Simbolo": symbol_value,
                "Nome": source_row.get("Nome", symbol_value),
                "Stato": status_value,
                "Motivi": (
                    "; ".join(all_reasons)
                    if all_reasons
                    else "Tutte le condizioni richieste sono soddisfatte"
                ),
                "Valuta": currency_value,
                "Capitalizzazione milioni": (
                    round(cap_value / 1_000_000, 2)
                    if cap_value is not None
                    else None
                ),
                "Crescita ricavi %": growth_value,
                "Margine operativo %": margin_value,
            })

        dashboard_df = pd.DataFrame(dashboard_rows)

        dashboard_passed = int(
            (dashboard_df["Stato"] == "Supera i filtri").sum()
        )
        dashboard_failed = int(
            (dashboard_df["Stato"] == "Non supera i filtri").sum()
        )
        dashboard_missing = int(
            (dashboard_df["Stato"] == "Dati insufficienti").sum()
        )

        st.caption(
            "Ricerca di riferimento: "
            + dashboard_saved.get("completed", "Non disponibile")
            + ". Le soglie sono quelle attualmente selezionate "
            "nel pannello dimensionale."
        )

        st.write(
            f"Valuta selezionata: {dashboard_currency}. "
            f"Dimensione: {dashboard_min_cap:,.0f}–"
            f"{dashboard_max_cap:,.0f} milioni. "
            f"Crescita minima: {dashboard_growth:.1f}%. "
            f"Profilo: {dashboard_margin_mode}."
        )

        count_col1, count_col2 = st.columns(2)
        count_col1.metric("Schede disponibili", len(dashboard_df))
        count_col2.metric("Superano i filtri", dashboard_passed)

        count_col3, count_col4 = st.columns(2)
        count_col3.metric("Non superano", dashboard_failed)
        count_col4.metric("Dati insufficienti", dashboard_missing)

        dashboard_fetch_errors = dashboard_saved.get("errors", [])
        if dashboard_fetch_errors:
            st.warning(
                f"Altre {len(dashboard_fetch_errors)} richieste "
                "hanno restituito errori: non sono incluse nei "
                "conteggi delle schede disponibili."
            )

        if dashboard_passed:
            st.success(
                f"{dashboard_passed} società corrispondono "
                "ai filtri. Sono candidate da studiare, "
                "non segnali di acquisto."
            )
        else:
            st.info(
                "Nessuna corrispondenza. La tabella seguente "
                "spiega perché, senza modificare le soglie."
            )

        def dashboard_status_style(value):
            styles = {
                "Supera i filtri": (
                    "background-color: #123A32; color: #B5F5DB;"
                ),
                "Non supera i filtri": (
                    "background-color: #3B2834; color: #FFD2DF;"
                ),
                "Dati insufficienti": (
                    "background-color: #3A321B; color: #FFE7A3;"
                ),
            }
            return styles.get(value, "")

        dashboard_styled = dashboard_df.style.apply(
            lambda column: column.map(dashboard_status_style),
            subset=["Stato"],
        )

        st.subheader("Motivo per ogni azienda")
        st.dataframe(
            dashboard_styled,
            hide_index=True,
            use_container_width=True,
        )

        dashboard_same_currency = dashboard_df[
            dashboard_df["Valuta"] == dashboard_currency
        ].copy()

        st.subheader("Grafici dei dati disponibili")
        st.caption(
            "Grafici limitati alla valuta selezionata. "
            "Le barre non sono previsioni né classifiche "
            "di qualità. Società senza il dato sono omesse."
        )

        growth_chart_df = dashboard_same_currency[
            ["Simbolo", "Crescita ricavi %"]
        ].dropna(subset=["Crescita ricavi %"])

        if not growth_chart_df.empty:
            st.write("Crescita ricavi riportata dal fornitore, in %")
            st.bar_chart(
                growth_chart_df.set_index("Simbolo"),
                color="#38BDF8",
            )

        margin_chart_df = dashboard_same_currency[
            ["Simbolo", "Margine operativo %"]
        ].dropna(subset=["Margine operativo %"])

        if not margin_chart_df.empty:
            st.write("Margine operativo riportato dal fornitore, in %")
            st.bar_chart(
                margin_chart_df.set_index("Simbolo"),
                color="#A78BFA",
            )

        cap_chart_df = dashboard_same_currency[
            ["Simbolo", "Capitalizzazione milioni"]
        ].dropna(subset=["Capitalizzazione milioni"])

        cap_chart_df = cap_chart_df[
            cap_chart_df["Capitalizzazione milioni"] > 0
        ]

        if not cap_chart_df.empty:
            st.write(
                "Capitalizzazione in milioni di "
                + str(dashboard_currency)
            )
            st.bar_chart(
                cap_chart_df.set_index("Simbolo"),
                color="#94A3B8",
            )

        st.download_button(
            "Esporta diagnosi completa",
            dashboard_df.to_csv(index=False).encode("utf-8-sig"),
            file_name="diagnosi_selezione.csv",
            mime="text/csv",
            key="dashboard_diagnosis_export_v4",
        )

        st.caption(
            "Dati insufficienti ha precedenza nello stato riepilogativo, "
            "ma la colonna Motivi mostra anche le condizioni "
            "già risultate negative. I periodi contabili possono "
            "differire tra società e tra indicatori."
        )
        st.divider()
st.header("Importa un universo di ricerca")

st.caption(
    "Carica una lista di simboli e analizzala per gruppi. "
    "La provenienza è dichiarata dall'utente, non verificata "
    "automaticamente. Importare un simbolo non conferma "
    "che sia valido o che l'azienda sia quotata oggi."
)

with st.expander("Formato del CSV e istruzioni", expanded=False):
    st.write(
        "La prima riga deve contenere la colonna Simbolo. "
        "Le altre colonne sono facoltative. "
        "Usa i simboli compatibili con il fornitore dell'app."
    )
    st.code(
        "Simbolo,Nome,Mercato\n"
        "AAPL,Apple,USA\n"
        "ENEL.MI,Enel,Italia\n"
        "SAP.DE,SAP,Germania",
        language="text",
    )
    st.caption(
        "Questi tre simboli sono soltanto un esempio di formato, "
        "non una lista di aziende emergenti."
    )

    st.download_button(
        "Scarica modello CSV",
        (
            "Simbolo,Nome,Mercato\n"
            "AAPL,Apple,USA\n"
            "ENEL.MI,Enel,Italia\n"
            "SAP.DE,SAP,Germania\n"
        ).encode("utf-8-sig"),
        file_name="modello_universo.csv",
        mime="text/csv",
        key="universe_template_v5",
    )

universe_source = st.text_input(
    "Fonte dell'elenco: nome e, se disponibile, indirizzo",
    placeholder="Esempio: elenco ufficiale della borsa consultato da me",
    key="universe_source_v5",
)

universe_date = st.date_input(
    "Data di riferimento dell'elenco",
    value=datetime.now(timezone.utc).date(),
    key="universe_date_v5",
)

universe_upload = st.file_uploader(
    "Carica il CSV dei simboli",
    type=["csv"],
    key="universe_upload_v5",
)

if st.button("Controlla e importa elenco", key="universe_import_v5"):
    if universe_upload is None:
        st.warning("Scegli prima un file CSV.")
    elif not universe_source.strip():
        st.warning(
            "Indica la provenienza dell'elenco. "
            "Se è una lista personale, scrivi 'Lista personale'."
        )
    elif universe_date > datetime.now(timezone.utc).date():
        st.warning("La data di riferimento non può essere futura.")
    elif universe_upload.size > 2_000_000:
        st.warning("Per questo modulo usa un CSV inferiore a 2 MB.")
    else:
        try:
            universe_upload.seek(0)
            imported_df = pd.read_csv(
                universe_upload,
                sep=None,
                engine="python",
                dtype=str,
                encoding="utf-8-sig",
                keep_default_na=False,
            )

            imported_df.columns = [
                str(column).strip()
                for column in imported_df.columns
            ]

            if "Simbolo" not in imported_df.columns:
                raise ValueError(
                    "Manca la colonna Simbolo. "
                    "Rinomina la colonna dei ticker e riprova."
                )

            if len(imported_df) > 5000:
                raise ValueError(
                    "Massimo 5.000 righe per questa versione."
                )

            accepted_symbols = []
            rejected_rows = []
            seen_symbols = set()
            duplicate_count = 0

            for row_number, raw_symbol in enumerate(
                imported_df["Simbolo"], start=2
            ):
                symbol = str(raw_symbol).strip().upper()

                if not symbol:
                    rejected_rows.append({
                        "Riga CSV": row_number,
                        "Valore": "",
                        "Motivo": "Simbolo vuoto",
                    })
                    continue

                if not re.fullmatch(
                    r"[A-Z0-9^][A-Z0-9.^=\-]{0,24}", symbol
                ):
                    rejected_rows.append({
                        "Riga CSV": row_number,
                        "Valore": symbol,
                        "Motivo": (
                            "Formato non supportato: "
                            "controllare il simbolo del fornitore"
                        ),
                    })
                    continue

                if symbol in seen_symbols:
                    duplicate_count += 1
                    continue

                seen_symbols.add(symbol)
                accepted_symbols.append(symbol)

            if not accepted_symbols:
                raise ValueError(
                    "Nessun simbolo con formato accettabile."
                )

            st.session_state["universe_imported_v5"] = {
                "symbols": accepted_symbols,
                "source": universe_source.strip(),
                "reference_date": universe_date.isoformat(),
                "imported_at": datetime.now(timezone.utc).strftime(
                    "%Y-%m-%d %H:%M UTC"
                ),
                "duplicates": duplicate_count,
                "rejected": rejected_rows,
            }

            st.success(
                "Elenco importato. Il controllo ha verificato "
                "solo il formato, non l'esistenza dei titoli."
            )

        except Exception as import_error:
            st.error(f"Importazione non riuscita: {import_error}")

saved_universe = st.session_state.get("universe_imported_v5")

if saved_universe:
    st.write("Fonte dichiarata:", saved_universe["source"])
    st.write(
        "Data di riferimento:",
        saved_universe["reference_date"],
    )
    st.caption(
        "Importazione: "
        + saved_universe["imported_at"]
        + ". L'elenco non viene aggiornato automaticamente."
    )

    universe_count = len(saved_universe["symbols"])
    universe_batches = [
        saved_universe["symbols"][start:start + 20]
        for start in range(0, universe_count, 20)
    ]

    universe_col1, universe_col2 = st.columns(2)
    universe_col1.metric("Simboli importati", universe_count)
    universe_col2.metric("Gruppi da analizzare", len(universe_batches))

    st.caption(
        f"Duplicati rimossi: {saved_universe['duplicates']}. "
        f"Righe scartate: {len(saved_universe['rejected'])}."
    )

    selected_batch_index = st.selectbox(
        "Scegli un gruppo",
        options=list(range(len(universe_batches))),
        format_func=lambda index: (
            f"Gruppo {index + 1} — "
            f"{len(universe_batches[index])} simboli"
        ),
        key="universe_batch_v5_" + str(universe_count),
    )

    selected_batch_text = ", ".join(
        universe_batches[selected_batch_index]
    )

    st.write("Simboli del gruppo selezionato:")
    st.code(selected_batch_text, language="text")

    st.info(
        "Copia questi simboli. Torna a 'Ricerca aziende', "
        "scegli 'Lista personale', incollali nel campo e premi "
        "'Cerca candidate nella lista'. "
        "L'importazione da sola non esegue lo screening."
    )

    st.download_button(
        "Esporta elenco pulito",
        pd.DataFrame({
            "Simbolo": saved_universe["symbols"],
            "Fonte dichiarata": saved_universe["source"],
            "Data riferimento": saved_universe["reference_date"],
        }).to_csv(index=False).encode("utf-8-sig"),
        file_name="universo_pulito.csv",
        mime="text/csv",
        key="universe_clean_export_v5",
    )

    if saved_universe["rejected"]:
        with st.expander("Righe da correggere"):
            st.dataframe(
                pd.DataFrame(saved_universe["rejected"]),
                hide_index=True,
                use_container_width=True,
            )

    st.warning(
        "L'elenco importato resta nella sessione corrente: "
        "conserva il CSV per poterlo ricaricare. "
        "Le ricerche successive sostituiscono i risultati "
        "della precedente: esporta ogni gruppo prima di "
        "passare al successivo. Nessuna scansione automatica "
        "di tutte le borse è attiva."
    )





