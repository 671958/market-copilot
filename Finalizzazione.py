import hashlib
import json
from datetime import datetime, timezone
import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf


def number(value):
    try:
        result = float(value)
        return result if np.isfinite(result) else None
    except (TypeError, ValueError):
        return None


def percent(value):
    result = number(value)
    return None if result is None else result * 100


def clean_json(value):
    if isinstance(value, dict):
        return {str(k): clean_json(v) for k, v in value.items()}
    if isinstance(value, list):
        return [clean_json(v) for v in value]
    if isinstance(value, (np.integer, np.floating)):
        return clean_json(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def stamp():
    return datetime.now(timezone.utc).isoformat(timespec='seconds')


def blank_archive():
    return {'version': 1, 'records': {}, 'errors': {}, 'notes': {}}


def validate_archive(data):
    if not isinstance(data, dict) or data.get('version') != 1:
        raise ValueError('Formato backup non riconosciuto')
    for key in ('records', 'errors', 'notes'):
        if not isinstance(data.get(key), dict) or len(data[key]) > 5000:
            raise ValueError('Sezione backup non valida: ' + key)
    for key, row in data['records'].items():
        if not isinstance(key, str) or not isinstance(row, dict):
            raise ValueError('Scheda non valida')
        if row.get('Simbolo') != key:
            raise ValueError('Simbolo incoerente nel backup')
    if not all(isinstance(v, str) for v in data['errors'].values()):
        raise ValueError('Errori backup non validi')
    if not all(isinstance(v, dict) for v in data['notes'].values()):
        raise ValueError('Note backup non valide')
    return data


@st.cache_data(ttl=21600, show_spinner=False)
def annual_accounts(symbol):
    ticker = yf.Ticker(symbol)
    income = ticker.income_stmt
    cash = ticker.cashflow
    labels = [
        (income, 'Total Revenue', 'Ricavi'),
        (income, 'Operating Income', 'Risultato operativo'),
        (income, 'Net Income', 'Utile netto'),
        (cash, 'Operating Cash Flow', 'Cassa operativa'),
        (cash, 'Capital Expenditure', 'Capex come riportato'),
    ]
    series = []
    for table, field, title in labels:
        if isinstance(table, pd.DataFrame) and field in table.index:
            values = table.loc[field]
            if isinstance(values, pd.DataFrame):
                values = values.iloc[0]
            values = pd.to_numeric(values, errors='coerce')
            values.index = pd.to_datetime(values.index).strftime('%Y-%m-%d')
            series.append(values.rename(title))
    if not series:
        raise ValueError('Nessun conto annuale disponibile')
    result = pd.concat(series, axis=1).sort_index()
    result.index.name = 'Fine periodo annuale'
    return result.dropna(how='all')


def render_workspace(history_fn, calculate_fn, fundamentals_fn, parse_fn):
    st.divider()
    st.header('Area di lavoro integrata')
    st.caption('Analisi per gruppi, archivio esportabile e dossier. Nessun ordine automatico.')
    if 'workspace_final_v1' not in st.session_state:
        st.session_state['workspace_final_v1'] = blank_archive()
    archive = st.session_state['workspace_final_v1']
    tabs = st.tabs(['1. Analizza', '2. Archivio', '3. Dossier', '4. Backup'])

    with tabs[0]:
        universe = st.session_state.get('universe_imported_v5', {})
        imported = universe.get('symbols', [])
        use_import = st.checkbox('Usa elenco importato', value=bool(imported), key='final_use_import')
        manual = st.text_area('Oppure inserisci simboli', 'AAPL, ENEL.MI, SAP.DE', key='final_manual')
        source = universe.get('source', 'Non dichiarata') if use_import else 'Lista personale'
        date = universe.get('reference_date', '') if use_import else ''
        symbols = list(dict.fromkeys(imported if use_import else parse_fn(manual)))
        symbols = [s for s in symbols if isinstance(s, str) and s.strip()]
        if len(symbols) > 5000:
            st.error('Massimo 5.000 simboli per elenco.')
            symbols = []
        if not symbols:
            st.info('Importa un elenco oppure disattiva la casella e inserisci simboli.')
        else:
            size = st.selectbox('Dimensione gruppo', [5, 10, 20], index=1, key='final_size')
            batches = [symbols[i:i+size] for i in range(0, len(symbols), size)]
            digest = hashlib.sha256(('|'.join(symbols) + str(size)).encode()).hexdigest()[:12]
            selected = st.selectbox('Gruppo da analizzare', range(len(batches)),
                                    format_func=lambda i: f'Gruppo {i+1}: {len(batches[i])} simboli',
                                    key='final_batch_' + digest)
            group = batches[selected]
            st.code(', '.join(group), language='text')
            force = st.checkbox('Rielabora anche le schede già archiviate', key='final_force')
            done = sum(s in archive['records'] for s in symbols)
            st.write(f'Schede archiviate per questo elenco: {done}/{len(symbols)}.')
            st.caption('Un gruppo per avvio. Nessuna scansione continua. I dati possono provenire dalla cache di 6 ore.')
            if st.button('Analizza questo gruppo', key='final_run', type='primary'):
                progress = st.progress(0)
                for i, symbol in enumerate(group):
                    if symbol not in archive['records'] or force or symbol in archive['errors']:
                        try:
                            info = fundamentals_fn(symbol)
                            if not isinstance(info, dict) or not info:
                                raise ValueError('Scheda vuota')
                            kind = info.get('quoteType')
                            if kind and kind != 'EQUITY':
                                raise ValueError('Tipo strumento escluso: ' + str(kind))
                            if not info.get('shortName') and number(info.get('marketCap')) is None:
                                raise ValueError('Identità societaria non confermata dal fornitore')
                            row = {
                                'Simbolo': symbol,
                                'Nome': info.get('longName') or info.get('shortName') or symbol,
                                'Tipo dichiarato': kind or 'Non disponibile',
                                'Settore': info.get('sector'),
                                'Valuta quotazione': info.get('currency'),
                                'Valuta bilancio': info.get('financialCurrency'),
                                'Capitalizzazione in milioni': None if number(info.get('marketCap')) is None else number(info.get('marketCap'))/1e6,
                                'Crescita ricavi %': percent(info.get('revenueGrowth')),
                                'Margine operativo %': percent(info.get('operatingMargins')),
                                'P/E storico': number(info.get('trailingPE')),
                                'Cassa operativa': number(info.get('operatingCashflow')),
                                'Sito dichiarato': info.get('website'),
                                'Attività secondo fornitore': info.get('longBusinessSummary'),
                                'Fonte elenco': source,
                                'Data elenco': date,
                                'Elaborazione UTC': stamp(),
                            }
                            try:
                                row.update(calculate_fn(history_fn(symbol)))
                                row['Copertura tecnica'] = 'Disponibile'
                            except Exception as exc:
                                row['Copertura tecnica'] = 'Non disponibile: ' + str(exc)
                            archive['records'][symbol] = clean_json(row)
                            archive['errors'].pop(symbol, None)
                        except Exception as exc:
                            archive['errors'][symbol] = str(exc)
                    progress.progress((i+1)/len(group))
                st.success('Gruppo terminato. Le schede precedenti sono conservate nella sessione.')
        if archive['errors']:
            with st.expander('Errori e richieste da riprovare'):
                st.json(archive['errors'])
            st.caption('Se un aggiornamento fallisce, l’eventuale vecchia scheda resta archiviata: verificarne la data.')

    with tabs[1]:
        if not archive['records']:
            st.info('Nessuna scheda archiviata.')
        else:
            df = pd.DataFrame(archive['records'].values())
            currencies = sorted({str(x) for x in df['Valuta quotazione'].dropna() if str(x)})
            choice = st.selectbox('Valuta (nessuna conversione)', ['Tutte'] + currencies, key='final_ccy')
            filtered = df if choice == 'Tutte' else df[df['Valuta quotazione'] == choice]
            minimum = st.number_input('Crescita minima ricavi %', value=10.0, key='final_growth')
            margin = st.selectbox('Margine operativo', ['Qualsiasi', 'Positivo', 'Negativo'], key='final_margin')
            filtered = filtered.copy()
            for col in ['Crescita ricavi %', 'Margine operativo %', 'Capitalizzazione in milioni']:
                filtered[col] = pd.to_numeric(filtered[col], errors='coerce')
            filtered = filtered[filtered['Crescita ricavi %'] >= minimum]
            if margin == 'Positivo':
                filtered = filtered[filtered['Margine operativo %'] > 0]
            elif margin == 'Negativo':
                filtered = filtered[filtered['Margine operativo %'] < 0]
            if choice != 'Tutte':
                cap = st.number_input('Capitalizzazione massima in milioni (0 = nessun limite)', min_value=0.0, value=10000.0, key='final_cap')
                if cap > 0:
                    filtered = filtered[(filtered['Capitalizzazione in milioni'] > 0) & (filtered['Capitalizzazione in milioni'] <= cap)]
            else:
                st.caption('Filtro monetario disabilitato con valute miste.')
            st.metric('Risultati sui dati disponibili', len(filtered))
            cols = ['Simbolo', 'Nome', 'Valuta quotazione', 'Capitalizzazione in milioni', 'Crescita ricavi %', 'Margine operativo %', 'Elaborazione UTC']
            st.dataframe(filtered[cols], hide_index=True, use_container_width=True)
            st.caption('I valori mancanti non soddisfano i filtri che li richiedono. Ordinamento non equivale a raccomandazione.')
            if not filtered.empty:
                st.bar_chart(filtered.set_index('Simbolo')[['Crescita ricavi %']], color='#38BDF8')
            st.download_button('Esporta selezione', filtered.to_csv(index=False).encode('utf-8-sig'), 'selezione_archivio.csv', 'text/csv', key='final_filtered')
            st.download_button('Esporta archivio completo CSV', df.to_csv(index=False).encode('utf-8-sig'), 'archivio_completo.csv', 'text/csv', key='final_all')

    with tabs[2]:
        if not archive['records']:
            st.info('Analizza almeno una società per aprire il dossier.')
        else:
            selected_symbol = st.selectbox('Società del dossier', sorted(archive['records']), key='final_dossier')
            record = archive['records'][selected_symbol]
            st.subheader(record.get('Nome', selected_symbol))
            st.caption('Dati sintetici: Yahoo Finance tramite yfinance. Non verificati su bilanci ufficiali.')
            st.write(record.get('Attività secondo fornitore') or 'Descrizione non disponibile.')
            site = record.get('Sito dichiarato')
            if isinstance(site, str) and site.startswith(('https://', 'http://')):
                st.link_button('Sito indicato dal fornitore: cerca Investor Relations', site)
            st.write('Valuta bilancio:', record.get('Valuta bilancio') or 'Non disponibile')
            if st.button('Recupera conti annuali disponibili', key='final_accounts'):
                try:
                    annual = annual_accounts(selected_symbol)
                    st.dataframe(annual, use_container_width=True)
                    if 'Ricavi' in annual:
                        st.line_chart(annual[['Ricavi']].dropna())
                    st.download_button('Esporta conti annuali', annual.to_csv().encode('utf-8-sig'), 'conti_annuali.csv', 'text/csv', key='final_annual_csv')
                    st.caption('Date di fine periodo, non date di pubblicazione. Non confronto automaticamente dati annuali e trimestrali. Nessuna conversione valutaria o rettifica settoriale.')
                except Exception as exc:
                    st.warning('Conti non disponibili: ' + str(exc))
            st.info('Tesi documentata: compilazione manuale. Non è una valutazione automatica di Buffett, né una verifica dei documenti.')
            labels = {
                'tesi': 'Perché potrebbe crescere? Specifica prodotto e domanda.',
                'vantaggio': 'Vantaggio competitivo e prove: non solo affermazioni.',
                'evento': 'Evento politico/geologico/industriale e nesso economico ipotizzato.',
                'fonti': 'Fonti ufficiali: URL, data del documento e passaggio rilevante.',
                'rischi': 'Controtesi: debito, diluizione, concorrenza, capitale necessario.',
                'prezzo': 'Valutazione: quali ipotesi giustificano il prezzo? Se manca, scrivi non valutato.',
                'invalidazione': 'Quali fatti smentirebbero la tesi?',
            }
            existing = archive['notes'].get(selected_symbol, {})
            with st.form('final_notes_' + selected_symbol):
                values = {k: st.text_area(label, value=str(existing.get(k, '')), key='final_note_'+selected_symbol+'_'+k) for k, label in labels.items()}
                if st.form_submit_button('Conserva note nella sessione'):
                    archive['notes'][selected_symbol] = {**values, 'saved_utc': stamp()}
                    st.success('Note conservate: esporta il backup per non perderle.')
            report = '\n\n'.join([selected_symbol, json.dumps(record, ensure_ascii=False, indent=2)] + [labels[k]+'\n'+str(archive['notes'].get(selected_symbol, {}).get(k, 'Non compilato')) for k in labels])
            st.download_button('Esporta dossier testuale', report.encode('utf-8'), 'dossier_'+selected_symbol+'.txt', 'text/plain', key='final_report')
            st.warning('Non sono calcolati valore intrinseco, affidabilità del vantaggio competitivo o effetto delle notizie. I campi non compilati restano non valutati.')

    with tabs[3]:
        st.warning('La sessione non è un archivio permanente. Scarica il backup prima di uscire. Non inserire credenziali o dati del conto.')
        backup = {'version': 1, 'records': archive['records'], 'errors': archive['errors'], 'notes': archive['notes']}
        st.download_button('Scarica backup completo JSON', json.dumps(clean_json(backup), ensure_ascii=False, allow_nan=False, indent=2).encode('utf-8'), 'market_copilot_backup.json', 'application/json', key='final_backup')
        uploaded = st.file_uploader('Ripristina un backup creato da questo modulo', type=['json'], key='final_restore_upload')
        if st.button('Importa backup e unisci all’archivio', key='final_restore'):
            try:
                if uploaded is None:
                    raise ValueError('Scegli prima il file')
                if uploaded.size > 10_000_000:
                    raise ValueError('Massimo 10 MB')
                restored = validate_archive(json.loads(uploaded.getvalue().decode('utf-8-sig')))
                for key in ('records', 'errors', 'notes'):
                    archive[key].update(restored[key])
                st.success('Backup importato. Per simboli uguali prevale il backup caricato. Le date dei dati restano originali.')
                st.rerun()
            except Exception as exc:
                st.error('Ripristino non riuscito: ' + str(exc))
        st.caption('Il backup contiene schede, errori e note. Non include elenco importato, conti annuali o archivio storico delle notizie. Usa solo file di provenienza fidata.')
