import streamlit as st
import surveydata_library.crosstabs as ct
import surveydata_library.stats as sstat
import surveydata_library.specific_usecases as suse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import regex as re


# --- Caching Functions ---
@st.cache_data
def load_data(map, data):
    df = pd.read_csv(data, low_memory=False)
    qmaps = pd.read_csv(map, encoding='latin1')
    return df, qmaps

@st.cache_data
def process_data(df, qmaps):
    # Multipunch Aggregation For Survey Data
    multipunch_cols = [col for col in df.columns if '.' in col]
    new_cols = pd.Series([x[:x.index('.')] for x in multipunch_cols]).unique()
    for question in new_cols:
        sub_questions = [x for x in multipunch_cols if x.startswith(question)]
        df[question] = df[sub_questions].apply(lambda row: row.tolist(), axis=1)

    # Load and preprocess question maps
    qmaps = qmaps.dropna(how='all')
    qmaps = qmaps[~qmaps[qmaps.columns[0]].str.contains('Type|Datatype')]
    qmaps = np.array(qmaps)

    maps = ct.genqmap(qmaps)
    maps['QBRAND'] = maps['QBRAND1']
    maps['QC1'] = maps['QC1_1001']
    maps['QAGE_RANGE'] = {'1':'18-24', '2':'25-34', '3':'35-44', '4':'45-54', '5':'55-64', '6':'65+'}
    quartermap = {
    'January': 'Q1',
    'February': 'Q1',
    'March': 'Q1',
    'April': 'Q2',
    'May': 'Q2',
    'June': 'Q2',
    'July': 'Q3',
    'August': 'Q3',
    'September': 'Q3',
    'October': 'Q4',
    'November': 'Q4',
    'December': 'Q4'
    }
    df['Quarter'] = df['QCALMONTH'].apply(lambda val: ct.getMap(maps, 'QCALMONTH')[str(int(val))])
    df['Quarter'] = df['Quarter'].str.extract(r'(\w+)', expand=False)
    df['Quarter'] = df['Quarter'].map(quartermap)
    return df, maps


# --- Main App ---
st.title("Historical Brand Survey Analysis")

# --- loqd necessary files ---
st.markdown('### Upload Necessary Files')
qmap_csv = st.file_uploader('Upload Question Map Here', type=['csv'])
data_csv = st.file_uploader('Upload Survey Data Here', type=['csv'])

# Load Data
df, qmaps = None, None
if data_csv is None or qmap_csv is None:
    st.warning("Please upload the necessary files.")
else:
    df, qmaps = load_data(qmap_csv, data_csv)
    df, maps = process_data(df, qmaps)

    # TODO: add weighting stuff later with respective funcs, for now leave it out
    
    # weightsmap = {
    #     'Gender': {'QGENDER': {'1': 49.0, '2': 51.0}},
    #     'Age': {'QAGE_RANGE': {'1': 11.8, '2': 17.6, '3': 16.8, '4': 15.8, '5': 16.4, '6': 21.6}}
    # }

    # --- UI: Brand Selection ---
    brands_selectable = ct.getMap(maps,'QBRAND')
    brand_keys = list(brands_selectable.keys())
    brand_values = list(brands_selectable.values())
    brands_selected = st.sidebar.multiselect("Select Brands", ['Select All'] + brand_values)

    # --- UI: pick a question ---
    q_option = st.sidebar.selectbox("Select a Question", ['QC1', 'QA1'])

    # --- UI: Filters ---
    st.sidebar.header("Filter Options")

    filters = []
    for q in ['QETHNICITY', 'QGENDER', 'QAGE_RANGE']:
        map_ = ct.getMap(maps,q)
        if map_:
            selected = st.sidebar.multiselect(f"Filter by {q}", list(map_.values()))
            filters += [f"{v}|{q}|{k}" for k, v in map_.items() if v in selected]

    # Create filter map
    filter_map = defaultdict(list)
    for entry in filters:
        parts = entry.split("|")
        question, code = parts[1], parts[2]
        filter_map[question].append(code)

    #TODO: check year and quarter selections later
    # --- UI: Year Selection ---
    years = sorted(df['QYEAR'].unique().astype(int))
    year_selected = st.sidebar.multiselect("Select Year(s)", years, default=years)

    # --- UI: Quarter Selection ---
    quarters = sorted(df['Quarter'].unique())
    quarter_selected = st.sidebar.multiselect("Select Quarter(s)", quarters, default=quarters)

    # --- UI: Box and Ranking ---
    box_option = st.sidebar.selectbox("Box Option", ['Top 2 Box', 'Bottom 2 Box', 'None'], index=2)
    rank_by = st.sidebar.checkbox("Rank by Percentage", value=False)
    time_series_brand_analysis = st.sidebar.checkbox("Show Time Series", value=False)


    # --- checking if any brand selected ---
    if 'Select All' in brands_selected:
        brand_codes = brand_keys
        brands_selected = brand_values
    else:   
        brand_codes = [brand_keys[brand_values.index(b)] for b in brands_selected]

    if len(brand_codes) == 0 or q_option is None:
        st.warning("To begin, please select at least one brand and one question.")
    else:       
        # --- Processing ---
        # TODO: is making a copy best practice? should we move this to a caching function? CURRENTLY REMOVED
        # my_df = df[['QC1', 'QBRAND', 'QCALMONTH', 'QYEAR', 'QGENDER', 'QETHNICITY', 'QAGE_RANGE']].copy()
        df.attrs['filters'] = {}
        df = ct.filter_by_dict(df, filter_map)
        if len(brand_codes) > 0:
            df = df[df['QBRAND'].isin(brand_codes)]
        df = ct.select_years(df, year_selected)
        '''
        TODO: check line below to make sure it works as intended
        '''
        df = df[df['Quarter'].isin(quarter_selected)]

        crosstab = ct.generate_crosstab(df[q_option], df['QBRAND'], maps, percentages=True)

        box_tab = None
        if box_option == 'Top 2 Box':
            crosstab, box_tab = ct.top_two(crosstab)
        elif box_option == 'Bottom 2 Box':
            crosstab, box_tab = ct.bot_two(crosstab)

        crosstab.columns = crosstab.columns.str.replace('All', 'Total')
        cols = crosstab.columns.tolist()
        crosstab = crosstab[[cols[-1]] + cols[:-1]]

        # --- Display Table ---
        st.subheader("Crosstab Output")
        st.dataframe(crosstab)

        # --- Ranking and Plot ---
        if rank_by and box_tab is not None:
            box_tab = ct.rank_over_percents(box_tab)
            st.subheader(f"{box_option} Ranked by Percentage")
            st.dataframe(box_tab)

            # Plot
            bargraph = (box_tab.drop(columns='All')).T
            bargraph = bargraph.iloc[:,0].apply(lambda x: float(re.match(r"(\d+)\s\|\s([\d\.]+)%", str(x)).group(2)))

            fig, ax = plt.subplots()
            bargraph.plot(kind='barh', ax=ax, color='teal')
            ax.set_title(f"Brands {box_option} Ranked by Percentage", fontsize=14)
            st.pyplot(fig)

        # --- Time Series ---
        if time_series_brand_analysis:
            # --- rendering, loading in UI ---
            which_brand = st.multiselect("Select Brand(s) for Time Series", brands_selected)
            brand_codes = [brand_keys[brand_values.index(b)] for b in which_brand]
            sub_categories = list(ct.getMap(maps, q_option).values())
            sub_select = st.sidebar.multiselect("Subquestion Select", sub_categories)

            # --- processing selections, generating plot ---
            df_brands = df[df['QBRAND'].isin(brand_codes)].copy()
            df_brands = df_brands.reset_index(drop=True)

            if len(sub_select) == 0:
                st.warning("Please select at least one subquestion.")
            else:
                fig2, ax2 = plt.subplots(figsize=(10, 4))
                for code in brand_codes:
                    df_brand = df_brands[df_brands['QBRAND'] == code].copy()
                    for sub_selected in sub_select:
                        #TODO: go over this method, probanly a better way to initially save the key code...
                        sub_map = ct.getMap(maps, q_option)  # {key: value}
                        sub_selected_key = [k for k, v in sub_map.items() if v == sub_selected][0]
                        if isinstance(df_brand[q_option].iloc[0], list):
                            # st.write('key:', sub_selected_key)
                            mask = df_brand[q_option].apply(lambda x: int(sub_selected_key) in x)
                            df_sub = df_brand[mask].copy()
                        else:
                            df_sub = df_brand[df_brand[q_option] == sub_selected_key].copy()
                        # st.write(df_brand.head())
                        df_sub = ct.sort_over_time(df_sub)

                        #TODO: change the periods to be quarters and year instead of qcalmonth
                        tseries = ct.generate_crosstab(df_sub[q_option], df_brand['QCALMONTH'], maps)
                        tseries = tseries.drop(columns=tseries.columns[-1])
                        tseries.loc['Total Answering'].plot(kind='line', ax=ax2, label=f"{sub_selected} ({brand_values[brand_keys.index(code)]})")

                st.subheader(f"Time Series - {', '.join(sub_select)}")
                ax2.set_title("Selected Subquestions Over Time")
                ax2.legend()
                plt.xticks(rotation=45, fontsize=8)
                plt.tight_layout()
                st.pyplot(fig2)
