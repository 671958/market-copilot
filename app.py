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
