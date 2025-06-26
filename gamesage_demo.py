# =========================================================
# GameSage Analytics – FULL Streamlit script
# Adds a looping arrow animation in front of each bullet on
# the Home page (no JSONDecodeError, no other logic touched)
# =========================================================

# ---------- 1. Imports ----------
import streamlit as st
from streamlit_option_menu import option_menu
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import base64
import streamlit.components.v1 as components
import numpy as np
from collections import Counter
from PIL import Image
from wordcloud import WordCloud
from textblob import TextBlob
from matplotlib.patches import Wedge
from scipy.signal import find_peaks

import matplotlib.patches as patches
base_theme = st.get_option("theme.base")
TEXT_COLOR = "#FFFFFF" if base_theme == "dark" else "#FFFFFF"
ICON_COLOR = "#FFFFFF" if base_theme == "dark" else "#FFFFFF"
# ---------- 1A.  Password flag ----------
if "dataset_auth" not in st.session_state:
    st.session_state.dataset_auth = False  
def safe_rerun():
    
    if hasattr(st, "rerun"):
        st.rerun()
    
    elif hasattr(st, "experimental_rerun"):
        st.experimental_rerun()
    
    else:
        st.info("Unlocked! Try refreshing the page ")

def set_background(image_file):
    with open(image_file, "rb") as f:
        data = f.read()
    encoded = base64.b64encode(data).decode()
    css = f"""
    <style>
    [data-testid="stAppViewContainer"] {{
        background-image: url("data:image/png;base64,{encoded}");
        background-size: cover;
        background-repeat: no-repeat;
        background-position: center;
        background-attachment: fixed;
    }}

    [data-testid="stHeader"] {{
        background: rgba(0,0,0,0);
    }}

    [data-testid="stToolbar"] {{
        right: 2rem;
        top: 2rem;
    }}

    /* Optional: make text white for better contrast */
    h1, h2, h3, h4, h5, h6, p, div, span {{
        color: white !important;
    }}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)
# NEW – Lottie support
from streamlit_lottie import st_lottie
import json, requests
COLOR_PALETTE = [
    # Original colors
      # Yellow-Orange
    "#60988C", 
    "#E3D19F", # Teal
    "#4E5D75",  # Muted Navy
    "#5C8DF6",  # Blue
    "#F4633A",  # Coral Red
]

# ---------- 2. Helper to fetch a Lottie *JSON* file ----------
@st.cache_data
def load_lottie(url: str):
    """Return a Python dict for a Lottie JSON animation."""
    try:
        resp = requests.get(url)
        if resp.status_code == 200:
            return resp.json()
    except Exception as e:
        st.warning(f"Failed to load Lottie animation: {e}")
    return None


menu_styles = {
    "container": {
        "padding": "0!important",
        "background-color": "#0e1117",     
        "overflow-x": "auto"
    },
    "nav-link": {
        "font-size": "16px",
        "white-space": "nowrap",
        "padding": "6px 14px",
        "color": TEXT_COLOR              
    },
    "icon": {"font-size": "18px", "color": ICON_COLOR},
    "nav-link-selected": {
        "background-color": "#b31010",
        "color": TEXT_COLOR               
    }
}
# ---------- 4. Navigation ----------
selected = option_menu(
    menu_title="GameSage Analytics",
    options=["Home", "Geospatial", "Visual Analysis", "Dataset", "Summary"],
    icons=["house", "geo-alt", "bar-chart", "database-add", "file-earmark-richtext"],
    default_index=0,
    orientation="horizontal",
    styles={           # normal nav colours
        "container": {"background-color": "#0e1117", "padding": "0!important"},
        "nav-link":   {"color": "#FFFFFF", "font-size": "16px"},
        "icon":       {"color": "#FFFFFF"},
        "menu_title": {"color": "#FFFFFF"},
        "nav-link-selected": {"background-color": "#b31010", "color": "#FFFFFF"},
    }
)
set_background("final_bg_try.png")
st.markdown(
    """
    <style>
    /* Force menu_title (navbar-brand) text to be white */
    .navbar .navbar-brand {
        color: #FFFFFF !important;
        font-weight: 600 !important;
    }

    /* Optional: improve alignment and spacing */
    .navbar {
        padding-left: 16px;
        padding-top: 6px;
        padding-bottom: 6px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ---------- 5. Data loading ----------
@st.cache_data
def load_data():
    sponsor_df = pd.read_csv("sponsor_detection.csv")
    audio_df = pd.read_csv("final_match_sponsor_data_colab (1).csv")
    df1 = pd.read_csv("cricket_shots.csv")
    df2 = pd.read_csv("IPL2k24_tweets_data.csv")
    df3 = pd.read_csv("stadium_boundaries.csv")
    df4 = pd.read_csv("engagement_peaks.csv")
    df6 = pd.read_csv("clean_tweet.csv")  
    return sponsor_df, audio_df, df1, df2, df3, df4, df6

try:
    sponsor_df, audio_df, df1, df2, df3, df4, df6 = load_data()
except Exception as e:
    st.error(f"Error loading data: {e}")

# =========================================================
# 6. Page routing
# =========================================================
if selected == "Home":
    st.markdown(
    """
    <h1 style='font-size: 2.3em; font-weight: bold;'>
        GameSage : <span style='color:white;'>Maximizing</span> Sponsor's <span style='color:white;'>ROI</span> in Franchise Cricket
    </h1>
    """,
    unsafe_allow_html=True
)
    
    st.markdown(
        """
        <h1 style='font-size: 1.5em; font-weight: bold;'>
            What We Offer  <span style='color:red;'>⇓</span>
         </h1>
         """,
        unsafe_allow_html=True  
    )   

    st.markdown(
        """
        <h1 style='font-size: 1.5em; font-weight: bold;'>
            <span style='color:white;'>►</span> Analysis of Sponsor Engagement & ROI Using Machine Learning and Computer Vision 
         </h1>
         """,
        unsafe_allow_html=True  
    )             
    
    st.markdown(
        """
        <h1 style='font-size: 1.5em; font-weight: bold;'>
            <span style='color:white;'>►</span> Identifying Blind Spots in Sponsor Visibility During Broadcasts
         </h1>
         """,
        unsafe_allow_html=True  
    )   

    st.markdown(
        """
        <h1 style='font-size: 1.5em; font-weight: bold;'>
            <span style='color:white;'>►</span> Geospatial Tagging Of Fans
         </h1>
         """,
        unsafe_allow_html=True  
    )   

    st.markdown(
        """
        <h1 style='font-size: 1.5em; font-weight: bold;'>
            <span style='color:white;'>►</span> Detecting Sponsor Logo Visibility During Peak Crowd Moments in IPL Videos by Conducting Audio Analysis and Using Computer Vision
         </h1>
         """,
        unsafe_allow_html=True  
    )   


    st.markdown(
        """
        <h1 style='font-size: 1.5em; font-weight: bold;'>
            <span style='color:white;'>►</span> Recommendations for Achieving Better ROI for Sponsors Through Predictive Mechanisms.
         </h1>
         """,
        unsafe_allow_html=True  
    )   





    

    

# ------------------ Geospatial ------------------
elif selected == "Geospatial":
    st.title("Geospatial Insights & Mapping Analysis")
    st.subheader("Map Targeting")
    with st.container():
        with st.expander("Power of Prediction & Analysis", expanded=True):
            st.markdown("""
Analyzing these hotspots lets organizations understand fan behavior, 
so they can target marketing campaigns and improve fan experiences 
in those specific areas.
""")
    components.iframe(
        "https://sponsor-map-33vc.vercel.app",
        height=600,
        width=1000,
        scrolling=True
    )

# --------------- Visual Analysis ----------------
elif selected == "Visual Analysis":
    st.title("Interactive Graphs and Data Visualizations:")

    with st.container():
        with st.expander("From Twitter(Physical Benchmark)", expanded=True):
            st.markdown("""
The graphs show spikes in tweet views and activity during key IPL 2024 moments. Higher peaks reflect viral tweets and major fan engagement days.
""")
        st.write("")
        st.write("")
        st.write("")
        st.write("")
        df2["tweet_created_at"] = pd.to_datetime(df2["tweet_created_at"])
        df2["time_rounded"] = df2["tweet_created_at"].dt.floor("T")
        view_over_time = df2.groupby("time_rounded")["tweet_view_count"].sum().reset_index()       
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(view_over_time["time_rounded"], view_over_time["tweet_view_count"], color=COLOR_PALETTE[3], marker="o")
        ax.set_xlabel("Time")
        ax.set_ylabel("Total Views")
        ax.set_title("Tweet Views Over Time")
        ax.grid(True)
        st.pyplot(fig)
        st.caption("Total views recorded between March and May 2024")
        st.write("")
        st.write("")
        st.write("")
        st.write("")
        st.write("")
        st.write("")
        st.write("")
        st.write("")
        st.write("")
        df2["tweet_created_at"] = pd.to_datetime(df2["tweet_created_at"])
        retweet_trend = df2.groupby(df2["tweet_created_at"].dt.date)["tweet_retweet_count"].sum()
        st.line_chart(retweet_trend)
        st.caption("Viral moments and key matches drove a surge in Twitter activity.")
    st.write("")
    st.write("")
    st.write("")
    st.write("")
    st.write("")
    st.write("")
    st.write("")
    st.write("")
    st.write("")
    st.write("")
    st.write("")
    with st.container():
        with st.expander("From Audio Peak Dataset (Missed Branding Opportunities)", expanded=True):
            st.markdown("""
The visuals show that sponsor visibility peaks at specific match moments, often aligning with high audio excitement. Most audio peaks cluster around a narrow score range, and sponsors like Dream11 and Kent were frequently visible during these engaging moments.
""")
        st.write("")
        st.write("")
        st.write("")
        st.write("")
        st.write("")
        st.write("")
        
        sponsor_count_df = audio_df[["TimestampFrameNumber", "VisibleSponsorsDuringPeak"]].copy()
        sponsor_count_df["SponsorList"] = sponsor_count_df["VisibleSponsorsDuringPeak"].str.split(", ")
        sponsor_count_df["SponsorCount"] = sponsor_count_df["SponsorList"].apply(
            lambda x: 0 if x == ["NoSponsorDetected"] else len(x)
        )

        # Filter for moments where at least 1 sponsor was visible
        filtered_df = sponsor_count_df[sponsor_count_df["SponsorCount"] > 0]

        # Bubble chart: Timestamp vs. SponsorCount (size = count)
        st.subheader("Sponsor Count at Key Match Moments")
        fig_bubble, ax_bubble = plt.subplots(figsize=(10, 4))
        ax_bubble.scatter(
            filtered_df["TimestampFrameNumber"],
            filtered_df["SponsorCount"],
            s=filtered_df["SponsorCount"] * 50,  # scale bubble size
            alpha=0.6,
            color=COLOR_PALETTE[4]
        )
        ax_bubble.set_xlabel("Timestamp (Frame Number)")
        ax_bubble.set_ylabel("Sponsor Count")
        ax_bubble.set_title("Number of Sponsors Visible at Key Match Moments")
        st.pyplot(fig_bubble)
        st.write("")
        st.write("")
        st.write("")
        st.write("")
        st.write("")
        st.write("")
        st.write("")
        st.write("")
        st.write("")
        st.write("")
        col5, col6 = st.columns(2)

        with col5:
            st.subheader("Audio Peak Score Distribution")
            fig_peak, ax_peak = plt.subplots()
            sns.histplot(audio_df["AudioPeakScore"], bins=20, kde=True, color=COLOR_PALETTE[4], ax=ax_peak)
            ax_peak.set_xlabel("AudioPeakScore")
            st.pyplot(fig_peak)
            st.write("")
            st.write("")
            st.write("")
            st.write("")
            st.write("")
            st.write("")
            st.write("")
        
        with col6:
            st.subheader("Sponsors Detected at Audio Peaks")
            exploded = audio_df["VisibleSponsorsDuringPeak"].str.split(", ").explode()
            peak_counts = exploded.value_counts().drop("NoSponsorDetected", errors="ignore")
            st.bar_chart(peak_counts,color=COLOR_PALETTE[4])
            st.write("")
            st.write("")
            st.write("")
            st.write("")
            st.write("")
            st.write("")
    with st.container():
        with st.expander("From Sponsor Detection Dataset (Missed Branding Opportunities)", expanded=True):
            st.markdown("""
This sponsor visibility analysis in cricket match frames. The bar charts highlight top sponsor appearances and where they were seen (like on jerseys or trousers).
""")
        st.write("")
        st.write("")
        st.write("")
        st.write("")

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Sponsor-wise Asset Count")
            sponsor_counts = sponsor_df["sponsor_name"].value_counts()
            st.bar_chart(sponsor_counts)
            st.write("")
            st.write("")
            st.write("")
            st.write("")

        with col2:
            st.subheader("Asset Type Distribution")
            asset_counts = sponsor_df["sponsor_asset_type"].value_counts()
            st.bar_chart(asset_counts)
            st.write("")
            st.write("")
            st.write("")
            st.write("")
            st.write("")
        st.subheader("Visibility Breakdown")
        fig_vis, ax_vis = plt.subplots()
        vis_counts = sponsor_df["sponsor_asset_visibility"].value_counts()
        PIE_COLORS = ["#60988C", 
        "#E3D19F", # Teal
        "#4E5D75",  # Muted Navy
        "#5C8DF6",  # Blue
        "#F4633A",]
        ax_vis.pie(vis_counts, labels=vis_counts.index, autopct='%1.1f%%', startangle=90, colors=PIE_COLORS)
        ax_vis.axis("equal")
        st.pyplot(fig_vis)
        st.write("")
        st.write("")
        st.write("")
        st.write("")
        st.write("")
        st.write("")
        st.write("")
        st.write("")
        st.write("")
    with st.container():
        with st.expander("From RCB VS PBKS(FINAL) BALL BY BALL DATASET (Power of Prediction and Analyis)", expanded=True):
            st.markdown("""
This chart and heatmap shows how often different cricket shot directions were played these hotspots helps predict where the ball will go, allowing for smart ad placement to get the most views.
""")
    st.write("")
    st.write("")
    st.write("")
    st.write("")
    shot_counts = df1['shot_direction'].value_counts().sort_index()
    shot_avg_runs = df1.groupby('shot_direction')['runs'].mean().loc[shot_counts.index]
    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Bar chart: counts
    sns.barplot(x=shot_counts.index, y=shot_counts.values, ax=ax1, palette=COLOR_PALETTE)
    ax1.set_xlabel("Shot Direction")
    ax1.set_ylabel("Delivery Count", color='blue')
    ax1.tick_params(axis='y', labelcolor='blue')
    ax1.set_xticklabels(shot_counts.index, rotation=45, ha='right')

    # Line chart: average runs
    ax2 = ax1.twinx()
    sns.lineplot(x=shot_avg_runs.index, y=shot_avg_runs.values, marker='o', color='red', ax=ax2)
    ax2.set_ylabel("Average Runs", color='red')
    ax2.tick_params(axis='y', labelcolor='red')

    ax1.set_title("Shot Direction Frequency and Average Runs") 
    st.pyplot(fig)

    coord_map = {
        # Straight shots
        "Straight": (0.5, 0.5),
        "Straight Long On": (0.5, 0.8),
        "Straight Mid Off": (0.5, 0.3),
        
        # On side (leg side) positions
        "Long On": (0.4, 0.9),
        "Mid-Wicket": (0.2, 0.6),
        "Mid-Wicket (Wide)": (0.15, 0.7),
        "Mid-Wicket (Caught)": (0.2, 0.6),
        "Mid Wicket": (0.2, 0.6),
        "Deep Mid-Wicket": (0.1, 0.8),
        "Deep Mid-Wicket (W)": (0.1, 0.8),
        "Square Leg": (0.1, 0.5),
        "Deep Squaring-Leg": (0.05, 0.7),
        "Long Leg": (0.1, 0.2),
        "Fine Leg": (0.15, 0.1),
        "Fine Leg (RHW)": (0.15, 0.1),
        "Fine Leg Covers": (0.2, 0.15),
        "Deep Leg": (0.05, 0.3),
        "Deep Leg Cut": (0.1, 0.4),
        "Deep Leg Cut (W)": (0.1, 0.4),
        "Deep Short Mid Off": (0.3, 0.7),
        
        # Off side positions
        "Long Off": (0.6, 0.9),
        "Long Off (LHW)": (0.6, 0.9),
        "Mid-Off": (0.7, 0.3),
        "Mid Off": (0.7, 0.3),
        "Mid-Off (Wide)": (0.75, 0.35),
        "Deep Mid Off": (0.8, 0.7),
        "Cover": (0.8, 0.4),
        "Covers": (0.8, 0.4),
        "Extra Covers": (0.85, 0.45),
        "Deep Covers": (0.9, 0.6),
        "Deep Cover Cover": (0.9, 0.6),
        "Point": (0.9, 0.3),
        "Forward Point": (0.85, 0.35),
        "Deep Forward Point": (0.95, 0.5),
        
        # Backward positions
        "Deep Backward Spin": (0.9, 0.2),
        
        # Scoop and unusual shots
        "Deceptive Scoop": (0.3, 0.9),
        
        # Pushes and defensive shots
        "Forward Push": (0.5, 0.4),
        "Defensive Push": (0.5, 0.4),
        
        # Dismissal types (center field for visualization)
        "Stumps (Bowled)": (0.5, 0.5),
        "Stumps (LBW)": (0.5, 0.5),
        "Stumps (Caught)": (0.5, 0.5),
        "Bowled Out/Loss Off": (0.5, 0.5),
        
        # Catches
        "Slicing Catch": (0.7, 0.6),
        "Slicing Catch (Long O": (0.6, 0.8),
        
        # Country (assuming this is a regional term for a specific field position)
        "Country": (0.3, 0.8),
        "Country (Wide)": (0.25, 0.85),
    }

    df1["coords"] = df1["shot_direction"].map(coord_map)
    # Use apply(pd.Series) to safely expand coords into x and y columns
    coords_df = df1["coords"].apply(pd.Series)
    coords_df.columns = ["x", "y"]
    df1 = df1.join(coords_df)

    agg = df1.groupby(["x", "y"])["runs"].agg("sum").reset_index()
    heatmap_data = agg.pivot(index="y", columns="x", values="runs").fillna(0)
    fig, ax = plt.subplots(figsize=(8, 8))
    sns.heatmap(
        heatmap_data,
        ax=ax,
        cmap=COLOR_PALETTE,
        cbar_kws={"label": "Total Runs"},
        annot=True,
        fmt='.0f',
        square=True
    )
    st.caption("Mapping shot frequency and runs to find scoring hotspots.")
    st.write("")
    st.write("")
    st.write("")
    st.write("")
    st.write("")
    st.write("")
    st.write("")
    st.write("")
    st.write("")
    st.write("")
    st.write("")
    st.write("")
    ax.set_title("Field-Position Heatmap of Runs")
    ax.set_xlabel("Field Position (X)")
    ax.set_ylabel("Field Position (Y)")
    st.pyplot(fig)
    st.caption("This heatmap shows where most runs are scored based on shot direction, helping brands plan ad placements in high action areas.")

# ------------------- Dataset --------------------
elif selected == "Dataset":
    # ---------- 2A. Password check ----------
    if not st.session_state.dataset_auth:
        st.subheader("🔒  Protected Section")
        pwd = st.text_input("Enter dataset password", type="password")
        if st.button("Unlock"):
            if pwd == "gamesage123":
                st.session_state.dataset_auth = True
                safe_rerun()
                      
            else:
                st.error("Wrong password, try again.")
        st.stop()

    
    st.title("Here are some sample of the datasets that we created: ")


    
    with st.expander("IPL 2024 Tweets Dataset", expanded=True):
        st.markdown("""
This dataset has thousands of tweets. It includes details like the tweet text, number of likes, retweets, views columns  you can quickly understand how people feel about the matches, how much attention each tweet got, and which players or teams were talked about the most.
""")
    
    st.dataframe(df2, use_container_width=True)
    
    st.markdown("""
<style>
/* Pura page ke liye text selection disable karna */
* {
    -webkit-user-select: none;
    -moz-user-select: none;
    -ms-user-select: none;
    user-select: none;
    -webkit-touch-callout: none;
    -o-user-select: none;
}

/* Specific elements ke liye bhi kar sakte hain */
.stMarkdown, .stText, .stDataFrame {
    -webkit-user-select: none;
    -moz-user-select: none;
    -ms-user-select: none;
    user-select: none;
}
</style>
""", unsafe_allow_html=True)
    st.markdown(
    """
    <style>
    [data-testid="stElementToolbar"] {
        display: none;
    }
    </style>
    """,
    unsafe_allow_html=True
)
    df2["tweet_created_at"] = pd.to_datetime(df2["tweet_created_at"])
    df2["time_rounded"] = df2["tweet_created_at"].dt.floor("T")  

    with st.container():
        with st.expander("Broadcast Audio-Peak / Sponsor Dataset", expanded=True):
            st.markdown("""
Frame-level audio-peak scores plus visible sponsors during peak moments.
Perfect for correlating brand exposure with crowd reaction.
""")
        st.dataframe(audio_df, use_container_width=True)

    with st.container():
        with st.expander("sponsor_detection(1)", expanded=True):
            st.markdown("""
This dataset has detailed info about sponsors seen matches, like match name, sponsor, and where it appeared on screen.It helps find which brands showed up most, where visibility was poor.
""")
        st.dataframe(sponsor_df, use_container_width=True)

    with st.container():
        with st.expander("Ball-by-Ball Data (RCB VS PBKS FINAL 2025)", expanded=True):
            st.markdown("""
This dataset has details of every ball like over, runs, bowler, batsman, and most importantly, shot direction. Using this, we can make charts and heatmaps to see where most shots go and plan ads placement better.
""")
        st.dataframe(df1, use_container_width=True)

    with st.container():
        with st.expander("Stadium Boundary Size", expanded=True):
            st.markdown("""
This dataset shows clear numbers about stadium sizes, like how long the boundaries are. It helps brands understand the physical space so they can compare stadiums and plan where to put their ads to get the most attention.
                        (NOTE! This Boundaries are measured from center)
""")
        st.dataframe(df3, use_container_width=True)

# ------------------- Summary --------------------
elif selected == "Summary":
    st.title("What We Achieved and How We Did It:")
    
    st.subheader("Found Blind Spots")
    with st.container():
        with st.expander(" A smart system that uses computer vision to finds spots in the stadium where ads are not clearly visible and helps sponsors to get the blind spots", expanded=True):
            img = Image.open("Final_blind_spots.png")
            st.image(img, width=img.width)
            st.write("")
            st.write("")
            st.write("")
            st.write("")
            st.write("")
            st.write("")


            # Load your images
# Make sure these files are in the correct path relative to your script
            image1 = Image.open("frame_13_47s.jpg")
            
            image2 = Image.open("frame_66_256s.jpg")

# Create two columns
            col1, col2 = st.columns(2)

# Display the first image in the first column
            with col1:
                st.image(image1, caption="Extracted Frames", use_container_width=True)

# Display the second image in the second column
            with col2:
                 st.image(image2, caption="Extracted Frames", use_container_width=True)


            image3 = Image.open("frame_17_64s.jpg")
            image4 = Image.open("frame_18_68s.jpg")
            st.write("")
            st.write("")
            st.write("")
            st.write("")
            st.write("")
            st.write("")
            
# Create two columns
            col1, col2 = st.columns(2)

# Display the first image in the first column
        with col1:
         st.image(image3, caption="Used Multiple Angles", use_container_width=True)

# Display the second image in the second column
        with col2:
         st.image(image4, caption="Used Multiple Angles", use_container_width=True)
            




        col1, col2 = st.columns(2)




        image5 = Image.open("Screenshot 2025-06-24 111459.png")
        image6 = Image.open("Screenshot 2025-06-24 111810.png")
        st.write("")
        st.write("")
        st.write("")
        st.write("")
        st.write("")
        st.write("")
# Create two columns
        col1, col2 = st.columns(2)

# Display the first image in the first column
        with col1:
         st.image(image5, caption="Marked Sponsors", use_container_width=True)

# Display the second image in the second column
        with col2:
         st.image(image6, caption="Marked Sponsors", use_container_width=True)




        image7 = Image.open("Screenshot 2025-06-24 122233.png")
        image8 = Image.open("Screenshot 2025-06-24 122343.png")
        st.write("")
        st.write("")
        st.write("")
        st.write("")
        st.write("")
        st.write("")
# Create two columns
        col1, col2 = st.columns(2)

# Display the first image in the first column
        with col1:
         st.image(image7, caption="Front & Back Banners", use_container_width=True)

# Display the second image in the second column
        with col2:
         st.image(image8, caption="Seats", use_container_width=True)




        image9 = Image.open("b1.jpg")
        image10 = Image.open("b2.jpg")
        st.write("")
        st.write("")
        st.write("")
        st.write("")
        st.write("")
        st.write("")
# Create two columns
        col1, col2 = st.columns(2)

# Display the first image in the first column
        with col1:
         st.image(image9, caption="Brightness Adjustments", use_container_width=True)

# Display the second image in the second column
        with col2:
         st.image(image10, caption="Brightness Adjustments", use_container_width=True)





    st.write("")
    st.write("")
    st.write("")
    st.write("")
    st.write("")
    st.write("")
    st.subheader("Fairplay Moments Detection")
    with st.container():
        with st.expander("We made a system that can spot good sportsmanship moments in cricket, like handshakes or helping another player. These moments are useful for sponsors to show their ads in a positive light.", expanded=True):
            img = Image.open("Screenshot_4K (2).png")
            st.image(img, width=img.width)
            st.write("")
            st.write("")  
            st.write("")
            st.write("")         
                    
            image1 = Image.open("handshake4.jpg")
            image2 = Image.open("hug3.jpg")        
            col1, col2 = st.columns(2)

# Display the first image in the first column
            with col1:
             st.image(image1, caption="Labeled Images with Handshakes", use_container_width=True)

# Display the second image in the second column
            with col2:
                st.image(image2, caption="Labeled Images with Hugs", use_container_width=True)

            

                    
            image3 = Image.open("image_search_1750792837020.jpg")
            image4 = Image.open("normal20.jpg")        
            col1, col2 = st.columns(2)

# Display the first image in the first column
            with col1:
             st.image(image3, caption="Labeled Images with Helping Moments", use_container_width=True)

# Display the second image in the second column
            with col2:
                st.image(image4, caption="Labeled Images with Normal Moments", use_container_width=True)



            st.write("")
            st.write("")
            st.write("")
            st.write("")
            st.write("")
            st.write("")
            image5 = Image.open("Screenshot 2025-06-25 010127.png")
            st.image(image5, caption="Extracted body poses for action recognition.", use_container_width=True)










    st.write("")
    st.write("")
    st.write("")
    st.write("")
    st.subheader("Fan Engagement Peak Detection")
    with st.container():
        with st.expander("We used an existing system to detect the most exciting moments in cricket match videos by analyzing crowd sounds.", expanded=True):
            img = Image.open("final_audio_map.png")
            st.image(img, width=img.width)


            st.write("")
            st.write("")
            st.write("")
            st.write("")
            st.caption("Extracted IPL Videos")
            st.video("clip.mp4")
            
            st.caption("Extracted Audio From The IPL Videos")

            audio_file = open('output_audio.wav', 'rb')
            st.audio(audio_file, format='audio/wav')
            st.write("")
            st.write("")   
            st.write("")
            st.write("")
            st.caption("Found Engagement Peaks")
            st.dataframe(df4, use_container_width=True)
            st.write("")
            st.write("")
            st.write("")
            st.caption("Plotted Excitement Graph and Then Marked Peaks on it.")

            peaks, _ = find_peaks(df4["audio_peak_score"], prominence=0.01)

            fig, ax = plt.subplots()

# Plot the main excitement curve
            ax.plot(df4["timestamp_sec"], df4["audio_peak_score"], label="Excitement Score", color="hotpink")

# Mark peaks on the graph
            ax.scatter(
            df4["timestamp_sec"].iloc[peaks],
            df4["audio_peak_score"].iloc[peaks],
            color="red",
            s=80,
            label="Peaks"
            )

            ax.set_xlabel("Time (s)")
            ax.set_ylabel("Excitement Score")
            ax.set_title("Excitement Over Time")
            ax.legend()

            st.pyplot(fig)



            st.write("")            
            st.write("")    
            st.write("")
            st.caption("Then Detected the Sponsors At the Peak Moments Using The Below Model That We Trained.")
            image7 = Image.open("val_batch0_labels.jpg")
            st.image(image7, caption="Sponsor's Detected", use_container_width=True)





    st.write("")
    st.write("")
    st.write("")
    st.write("")

    st.subheader("Sponsor Detection From Images And Video Frames")
    with st.container():
        with st.expander("We found the most exciting moments in IPL match videos using crowd sounds, then used computer vision to detect which sponsors appeared on screen during those moments. This helps brands understand where their logos are seen when fans are most engaged.", expanded=True):
            img = Image.open("final_sponsor_detection_map.png")
            st.image(img, width=img.width)
            st.write("")
            st.write("")
            st.write("")
            st.write("")
            st.write("")
            st.write("")
            image1 = Image.open("image_37_c0ce8732-42a6-414e-9ac0-7b6325011634.jpg")
            image2 = Image.open("image_39_4bfa6842-f74c-4f5a-99b2-7bd5e51e698d.jpg")        
            col1, col2 = st.columns(2)

# Display the first image in the first column
            with col1:
             st.image(image1, caption="Extracted Images from Peak Frames and Various Sources", use_container_width=True)

# Display the second image in the second column
            with col2:
                st.image(image2, caption="Extracted Images from Peak Frames and Various Sources", use_container_width=True)

            image3 = Image.open("image_93_ba30d224-1f30-4357-b05e-de1faa8e607a.jpg")
            image4 = Image.open("image_162_05ee7c21-a8d4-4a5f-be12-2413c03c6838.jpg")        
            col1, col2 = st.columns(2)                  


            with col1:
             st.image(image3, caption="Extracted Images from Peak Frames and Various Sources", use_container_width=True)

# Display the second image in the second column
            with col2:
                st.image(image4, caption="Extracted Images from Peak Frames and Various Sources", use_container_width=True)


            st.write("")
            st.write("")
            st.write("")
            st.write("")
            image5 = Image.open("train_batch2.jpg")
            image6 = Image.open("train_batch200.jpg")        
            col1, col2 = st.columns(2)                  


            with col1:
             st.image(image5, caption="Labeled Images With The Following Labels", use_container_width=True)

# Display the second image in the second column
            with col2:
                st.image(image6, caption="Labeled Images With The Following Labels", use_container_width=True)



        

            labels = [
   
"Acko_helmet_logo_clear",
"Acko_helmet_logo_partially_visible",
"Acko_helmet_logo_blurry",
"Acko_helmet_logo_obstructed",
"Acko_cap_logo_clear",
"Acko_cap_logo_partially_visible",
"Acko_cap_logo_blurry",
"Acko_cap_logo_obstructed",
"AllSeasons_jersey_chest_logo_clear",
"AllSeasons_jersey_chest_logo_partially_visible",
"AllSeasons_jersey_chest_logo_blurry",
"AllSeasons_jersey_chest_logo_obstructed",
"AstralPipes_jersey_shoulder_logo_clear",
"AstralPipes_jersey_shoulder_logo_partially_visible",
"AstralPipes_jersey_shoulder_logo_blurry",
"AstralPipes_jersey_shoulder_logo_obstructed",
"AvonCycles_trousers_logo_clear",
"AvonCycles_trousers_logo_partially_visible",
"AvonCycles_trousers_logo_blurry",
"AvonCycles_trousers_logo_obstructed",
"BKT_jersey_back_logo_clear",
"BKT_jersey_back_logo_partially_visible",
"BKT_jersey_back_logo_blurry",
"BKT_jersey_back_logo_obstructed",
"BKT_boundary_rope_clear",
"BKT_boundary_rope_partially_visible",
"BKT_boundary_rope_blurry",
"BKT_boundary_rope_obstructed",
"Dazzler_cheerleader_top_logo_clear",
"Dazzler_cheerleader_top_logo_partially_visible",
"Dazzler_cheerleader_top_logo_blurry",
"Dazzler_cheerleader_top_logo_obstructed",
"Dazzler_cheerleader_skirt_logo_clear",
"Dazzler_cheerleader_skirt_logo_partially_visible",
"Dazzler_cheerleader_skirt_logo_blurry",
"Dazzler_cheerleader_skirt_logo_obstructed",
"Dazzler_foreground_banner_clear",
"Dazzler_foreground_banner_partially_visible",
"Dazzler_foreground_banner_blurry",
"Dazzler_foreground_banner_obstructed",
"DPWorld_jersey_back_logo_clear",
"DPWorld_jersey_back_logo_partially_visible",
"DPWorld_jersey_back_logo_blurry",
"DPWorld_jersey_back_logo_obstructed",
"DPWorld_helmet_logo_clear",
"DPWorld_helmet_logo_partially_visible",
"DPWorld_helmet_logo_blurry",
"DPWorld_helmet_logo_obstructed",
"Dream11_jersey_chest_logo_clear",
"Dream11_jersey_chest_logo_partially_visible",
"Dream11_jersey_chest_logo_blurry",
"Dream11_jersey_chest_logo_obstructed",
"EcoLink_cap_logo_clear",
"EcoLink_cap_logo_partially_visible",
"EcoLink_cap_logo_blurry",
"EcoLink_cap_logo_obstructed",
"EcoLink_helmet_back_logo_clear",
"EcoLink_helmet_back_logo_partially_visible",
"EcoLink_helmet_back_logo_blurry",
"EcoLink_helmet_back_logo_obstructed",
"Encalm_trousers_logo_clear",
"Encalm_trousers_logo_partially_visible",
"Encalm_trousers_logo_blurry",
"Encalm_trousers_logo_obstructed",
"Equitas_helmet_logo_clear",
"Equitas_helmet_logo_partially_visible",
"Equitas_helmet_logo_blurry",
"Equitas_helmet_logo_obstructed",
"Freemans_cap_logo_clear",
"Freemans_cap_logo_partially_visible",
"Freemans_cap_logo_blurry",
"Freemans_cap_logo_obstructed",
"GMR_jersey_sleeve_logo_clear",
"GMR_jersey_sleeve_logo_partially_visible",
"GMR_jersey_sleeve_logo_blurry",
"GMR_jersey_sleeve_logo_obstructed",
"GMR_jersey_chest_logo_clear",
"GMR_jersey_chest_logo_partially_visible",
"GMR_jersey_chest_logo_blurry",
"GMR_jersey_chest_logo_obstructed",
"GMR_helmet_logo_clear",
"GMR_helmet_logo_partially_visible",
"GMR_helmet_logo_blurry",
"GMR_helmet_logo_obstructed",
"HeroFincorp_jersey_chest_logo_clear",
"HeroFincorp_jersey_chest_logo_partially_visible",
"HeroFincorp_jersey_chest_logo_blurry",
"HeroFincorp_jersey_chest_logo_obstructed",
"HeroFincorp_jersey_sleeve_logo_clear",
"HeroFincorp_jersey_sleeve_logo_partially_visible",
"HeroFincorp_jersey_sleeve_logo_blurry",
"HeroFincorp_jersey_sleeve_logo_obstructed",
"HeroFincorp_helmet_logo_clear",
"HeroFincorp_helmet_logo_partially_visible",
"HeroFincorp_helmet_logo_blurry",
"HeroFincorp_helmet_logo_obstructed",
"Jio_jersey_shoulder_logo_clear",
"Jio_jersey_shoulder_logo_partially_visible",
"Jio_jersey_shoulder_logo_blurry",
"Jio_jersey_shoulder_logo_obstructed",
"Jio_cap_logo_clear",
"Jio_cap_logo_partially_visible",
"Jio_cap_logo_blurry",
"Jio_cap_logo_obstructed",
"JioHotstar_clear",
"JioHotstar_partially_visible",
"JioHotstar_blurry",
"JioHotstar_obstructed",
"JSWPaints_helmet_logo_clear",
"JSWPaints_helmet_logo_partially_visible",
"JSWPaints_helmet_logo_blurry",
"JSWPaints_helmet_logo_obstructed",
"JSWPaints_cap_logo_clear",
"JSWPaints_cap_logo_partially_visible",
"JSWPaints_cap_logo_blurry",
"JSWPaints_cap_logo_obstructed",
"JSWPaints_jersey_chest_logo_clear",
"JSWPaints_jersey_chest_logo_partially_visible",
"JSWPaints_jersey_chest_logo_blurry",
"JSWPaints_jersey_chest_logo_obstructed",
"KentMineralRO_jersey_chest_logo_clear",
"KentMineralRO_jersey_chest_logo_partially_visible",
"KentMineralRO_jersey_chest_logo_blurry",
"KentMineralRO_jersey_chest_logo_obstructed",
"kshema_trousers_logo_clear",
"kshema_trousers_logo_partially_visible",
"kshema_trousers_logo_blurry",
"kshema_trousers_logo_obstructed",
"LivPure_trousers_logo_clear",
"LivPure_trousers_logo_partially_visible",
"LivPure_trousers_logo_blurry",
"LivPure_trousers_logo_obstructed",
"motorola_trousers_logo_clear",
"motorola_trousers_logo_partially_visible",
"motorola_trousers_logo_blurry",
"motorola_trousers_logo_obstructed",
"Puma_jersey_sleeve_logo_clear",
"Puma_jersey_sleeve_logo_partially_visible",
"Puma_jersey_sleeve_logo_blurry",
"Puma_jersey_sleeve_logo_obstructed",
"RayzonSolar_trousers_logo_clear",
"RayzonSolar_trousers_logo_partially_visible",
"RayzonSolar_trousers_logo_blurry",
"RayzonSolar_trousers_logo_obstructed",
"SimpoloCeramics_jersey_shoulder_logo_clear",
"SimpoloCeramics_jersey_shoulder_logo_partially_visible",
"SimpoloCeramics_jersey_shoulder_logo_blurry",
"SimpoloCeramics_jersey_shoulder_logo_obstructed",
"Spinner_Sports_Drink_clear",
"Spinner_Sports_Drink_partially_visible",
"Spinner_Sports_Drink_blurry",
"Spinner_Sports_Drink_obstructed",
"Torrent_Group_jersey_chest_logo_clear",
"Torrent_Group_jersey_chest_logo_partially_visible",
"Torrent_Group_jersey_chest_logo_blurry",
"Torrent_Group_jersey_chest_logo_obstructed",
"Valvoline_jersey_sleeve_logo_clear",
"Valvoline_jersey_sleeve_logo_partially_visible",
"Valvoline_jersey_sleeve_logo_blurry",
"Valvoline_jersey_sleeve_logo_obstructed",
"AllSeasons_jersey_sleeve_logo_clear",
"AllSeasons_jersey_sleeve_logo_partially_visible",
"AllSeasons_jersey_sleeve_logo_blurry",
"AllSeasons_jersey_sleeve_logo_obstructed",

]
    st.write("")
    st.write("")
    st.write("")
    st.write("")
    st.write("")
    st.write("")
# Show total count
    st.write(f"Total Labels: {len(labels)}")

# Show in a table
    st.dataframe({"Label": labels})

    st.write("")
    st.write("")
    st.write("")
    st.write("")
    st.write("")

    st.caption("Then Trained a Model to Detect the Sponsors in the Images and Video Frames.")
    image10 = Image.open("val_batch0_pred.jpg")
    st.image(image10, caption="Our Model Predictions", use_container_width=True)

    st.write("")
    st.write("")        
    st.write("")
    st.write("")
    st.write("")
    st.subheader("Geospatial analysis of Fan Engagement")
    with st.container():
        with st.expander("We analyzed thousands of IPL tweets and maps where cricket fans are most active. Based on this data, it created small geographic zones and gave sponsor-wise suggestions for delivery apps, local businesses, and service providers. The goal was to help brands target areas with high fan activity more effectively.", expanded=True):
            img = Image.open("geo_map_final.png")
            st.image(img, width=img.width)
            st.write("")
            st.write("")
            st.write("")
            st.write("")
            st.write("")
            st.caption("Here are Some Samples of the Tweets We Collected and Then GeoTagged Them")
            
            
            st.dataframe(df6, use_container_width=True)
            st.write("")
            st.write("")
            st.write("")
            st.write("")
            st.caption("Then We Created a Map of IPL Fans' Locations and Engagement Hotspots")
            image13 = Image.open("Screenshot 2025-06-25 092720.png")
            st.image(image13, caption="", use_container_width=True)
            st.write("")
            st.write("")
            st.write("")
            st.write("")
            image14 = Image.open("Screenshot 2025-06-25 103201.png")
            st.image(image14, caption="", use_container_width=True)
# =========================================================
# End of script
# =========================================================
