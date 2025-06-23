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
import streamlit.components.v1 as components
import numpy as np
from collections import Counter
from PIL import Image
from wordcloud import WordCloud
from textblob import TextBlob
from matplotlib.patches import Wedge
import matplotlib.patches as patches


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

# Use a working Lottie JSON URL
ARROW = load_lottie("https://lottie.host/ab24ce1c-6e60-4c55-9e3e-fba77729ae19/qD8Ogwhw0w.json")
DATASET_ANIM = load_lottie(        # NEW – dataset page icon
    "https://lottie.host/752058be-30c2-4f90-8d07-40899c8005a3/Sf95jOePWK.json")  


# Bullet function with error handling
def bullet(text: str):
    col_icon, col_txt = st.columns([2, 25])
    with col_icon:
        if ARROW is not None:
            st_lottie(ARROW, width=40, height=40, key=text)
        else:
            st.write("→")  # Fallback to text arrow
    with col_txt:
        st.subheader(text)

# ---------- 3. Menu styling ----------
menu_styles = {
    "container": {
        "padding": "0!important",
        "background-color": "#0e1117",
        "overflow-x": "auto"
    },
    "nav-link": {
        "font-size": "16px",
        "white-space": "nowrap",
        "padding": "6px 14px"
    },
    "icon": {"font-size": "18px"},
    "nav-link-selected": {"background-color": "#b31010"}
}

# ---------- 4. Navigation ----------
selected = option_menu(
    menu_title="GameSage Analytics",
    options=["Home", "Geospatial", "Visual Analysis", "Dataset", "Summary"],
    icons=["house", "geo-alt", "bar-chart", "database-add", "file-earmark-richtext"],
    default_index=0,
    orientation="horizontal",
    styles=menu_styles
)

# ---------- 5. Data loading ----------
@st.cache_data
def load_data():
    sponsor_df = pd.read_csv("sponsor_detection.csv")
    audio_df = pd.read_csv("final_match_sponsor_data_colab (1).csv")
    df1 = pd.read_csv("cricket_shots.csv")
    df2 = pd.read_csv("IPL2k24_tweets_data.csv")
    df3 = pd.read_csv("stadium_boundaries.csv")
    return sponsor_df, audio_df, df1, df2, df3

try:
    sponsor_df, audio_df, df1, df2, df3 = load_data()
except Exception as e:
    st.error(f"Error loading data: {e}")

# =========================================================
# 6. Page routing
# =========================================================
if selected == "Home":
    st.markdown(
    """
    <h1 style='font-size: 2.3em; font-weight: bold;'>
        GameSage : <span style='color:red;'>Maximizing</span> Sponsor's <span style='color:red;'>ROI</span> in Franchise Cricket
    </h1>
    """,
    unsafe_allow_html=True
)
    
    st.markdown(
        """
        <h1 style='font-size: 1.5em; font-weight: bold;'>
            What We Bring  <span style='color:red;'>⇓</span>
         </h1>
         """,
        unsafe_allow_html=True  
    )   

    st.markdown(
        """
        <h1 style='font-size: 1.5em; font-weight: bold;'>
            <span style='color:red;'>►</span> Predictive Analysis of Sponsor Engagement & ROI Using Machine Learning and Computer Vision:  
         </h1>
         """,
        unsafe_allow_html=True  
    )             
    
    st.markdown(
        """
        <h1 style='font-size: 1.5em; font-weight: bold;'>
            <span style='color:red;'>►</span> Identifying Blind Spots in Sponsor Visibility During Broadcasts:  
         </h1>
         """,
        unsafe_allow_html=True  
    )   

    st.markdown(
        """
        <h1 style='font-size: 1.5em; font-weight: bold;'>
            <span style='color:red;'>►</span> Geospatial Tagging Of Fans:  
         </h1>
         """,
        unsafe_allow_html=True  
    )   

    st.markdown(
        """
        <h1 style='font-size: 1.5em; font-weight: bold;'>
            <span style='color:red;'>►</span> Detecting Sponsor Logo Visibility During Peak Crowd Moments in IPL Videos Using Audio and Computer Vision:  
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

    df2["tweet_created_at"] = pd.to_datetime(df2["tweet_created_at"])
    retweet_trend = df2.groupby(df2["tweet_created_at"].dt.date)["tweet_retweet_count"].sum()
    st.line_chart(retweet_trend)

    with st.container():
        with st.expander("From Audio Peak Dataset (Missed Branding Opportunities)", expanded=True):
            st.markdown("""
The visuals show that sponsor visibility peaks at specific match moments, often aligning with high audio excitement. Most audio peaks cluster around a narrow score range, and sponsors like Dream11 and Kent were frequently visible during these engaging moments.
""")
            
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

    col5, col6 = st.columns(2)

    with col5:
        st.subheader("Audio Peak Score Distribution")
        fig_peak, ax_peak = plt.subplots()
        sns.histplot(audio_df["AudioPeakScore"], bins=20, kde=True, color=COLOR_PALETTE[4], ax=ax_peak)
        ax_peak.set_xlabel("AudioPeakScore")
        st.pyplot(fig_peak)

    with col6:
        st.subheader("Sponsors Detected at Audio Peaks")
        exploded = audio_df["VisibleSponsorsDuringPeak"].str.split(", ").explode()
        peak_counts = exploded.value_counts().drop("NoSponsorDetected", errors="ignore")
        st.bar_chart(peak_counts,color=COLOR_PALETTE[4])

    with st.container():
        with st.expander("From Sponsor Detection Dataset (Missed Branding Opportunities)", expanded=True):
            st.markdown("""
This sponsor visibility analysis in cricket match frames. The bar charts highlight top sponsor appearances and where they were seen (like on jerseys or trousers).
""")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Sponsor-wise Asset Count")
        sponsor_counts = sponsor_df["sponsor_name"].value_counts()
        st.bar_chart(sponsor_counts)

    with col2:
        st.subheader("Asset Type Distribution")
        asset_counts = sponsor_df["sponsor_asset_type"].value_counts()
        st.bar_chart(asset_counts)

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

    with st.container():
        with st.expander("From RCB VS PBKS(FINAL) BALL BY BALL DATASET (Power of Prediction and Analyis)", expanded=True):
            st.markdown("""
This chart and heatmap shows how often different cricket shot directions were played these hotspots helps predict where the ball will go, allowing for smart ad placement to get the most views.
""")

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
    ax.set_title("Field-Position Heatmap of Runs")
    ax.set_xlabel("Field Position (X)")
    ax.set_ylabel("Field Position (Y)")
    st.pyplot(fig)

# ------------------- Dataset --------------------
elif selected == "Dataset":
    # ---------- 2A. Password check ----------
    if not st.session_state.dataset_auth:
        st.subheader("🔒  Protected Section")
        pwd = st.text_input("Enter dataset password", type="password")
        if st.button("Unlock"):
            if pwd == "gamesage123":          # <-- apna secret rakhna
                st.session_state.dataset_auth = True
                safe_rerun()
                      # refresh to show content
            else:
                st.error("Wrong password, try again.")
        st.stop()

    # ek chhota icon + text ka 2-column layout
    col_icon, col_text = st.columns([2, 4])
    with col_icon:
        st_lottie(
        DATASET_ANIM,
        height=230,          # pehle 140 tha
        width=230,
        quality="high",      # crisp lines & anti-aliasing
        loop=True,
        key="dataset_anim"
    )


    with col_text:
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
        with st.expander("Broadcast Audio-Peak / Sponsor Dataset", expanded=False):
            st.markdown("""
Frame-level audio-peak scores plus visible sponsors during peak moments.
Perfect for correlating brand exposure with crowd reaction.
""")
        st.dataframe(audio_df, use_container_width=True)

    with st.container():
        with st.expander("sponsor_detection(1)", expanded=False):
            st.markdown("""
This dataset has detailed info about sponsors seen matches, like match name, sponsor, and where it appeared on screen.It helps find which brands showed up most, where visibility was poor.
""")
        st.dataframe(sponsor_df, use_container_width=True)

    with st.container():
        with st.expander("Ball-by-Ball Data (RCB VS PBKS FINAL 2025)", expanded=False):
            st.markdown("""
This dataset has details of every ball like over, runs, bowler, batsman, and most importantly, shot direction. Using this, we can make charts and heatmaps to see where most shots go and plan ads placement better.
""")
        st.dataframe(df1, use_container_width=True)

    with st.container():
        with st.expander("Stadium Boundary Size", expanded=False):
            st.markdown("""
This dataset shows clear numbers about stadium sizes, like how long the boundaries are. It helps brands understand the physical space so they can compare stadiums and plan where to put their ads to get the most attention.
""")
        st.dataframe(df3, use_container_width=True)

# ------------------- Summary --------------------
elif selected == "Summary":
    st.title("What We Achieved and How We Did It:")
    
    st.subheader("Found Blind Spots")
    with st.container():
        with st.expander(" A smart system that uses computer vision to finds spots in the stadium where ads are not clearly visible and helps sponsors to get the blind spots", expanded=True):
            st.image(
        "diagram-export-6-23-2025-4_21_53-PM.png",               
        use_column_width=True,                      
        caption="Blind-spot detection workflow"
    )  
          

    st.subheader("Fairplay Moments Detection")
    with st.container():
        with st.expander("We made a system that can spot good sportsmanship moments in cricket, like handshakes or helping another player. These moments are useful for sponsors to show their ads in a positive light.", expanded=False):
            st.markdown("""
1. Pose Detection:
We used a tool called YOLOv8 to find body points (like head, arms, legs) from images taken from cricket matches.

2. Dataset Creation:
We collected and labeled 52 images:

10 handshakes

11 hugs

10 helping moments (like tying shoelaces)

21 normal (non-fairplay) moments

3. Feature Extraction:
From these images, we got 34 body keypoints per person (like joints' X and Y positions).

4. Model Training:
We used a Random Forest model to learn from these body positions and guess if the moment is fairplay or not.
""")
            
    st.subheader("Fan Engagement Peak Detection")
    with st.container():
        with st.expander("We used an existing system to detect the most exciting moments in cricket match videos by analyzing crowd sounds.", expanded=False):
            st.markdown("""
1. Audio Extraction:
We used MoviePy to pull the sound from a match video and save it as an audio file.

2. Sound Detection with YAMNet:
We used a pre-trained model called YAMNet to listen to short pieces of the audio and figure out what kind of sound is happening—like cheering, clapping, or crowd noise.

3. Engagement Scoring:
We focused only on sounds that show fan excitement. We gave each moment a score based on how loud or exciting the crowd sounded.

4. Finding Exciting Moments:
We used a method to detect where these excitement scores suddenly peak, which usually matches moments like boundaries or wickets.

5. Visualizing Peaks:
We plotted a graph to show excitement levels over time and marked the peaks with red dots. These peaks can be linked back to the video to know exactly when exciting moments happened.
""")

    st.subheader("Sponsor Detection From Video Frames")
    with st.container():
        with st.expander("We found the most exciting moments in IPL match videos using crowd sounds, then used computer vision to detect which sponsors appeared on screen during those moments. This helps brands understand where their logos are seen when fans are most engaged.", expanded=False):
            st.markdown("""
1: Collected Media
We collected photos from the IPL website and videos from various online platforms.

2: Detected Exciting Moments
We used a model called YAMNet to listen for sounds like cheering, clapping, or shouting and find the moments when fan excitement was highest.

3: Captured Key Frames
We took snapshots from the video at those exciting moments.

4: Labeled Sponsor Images
We labeled over 500 sponsor images  to train the model.

5: Trained YOLOv8 Model
We trained a YOLOv8 model to recognize sponsors in the images.

6: Detected Sponsors in Exciting Moments
We used the trained model to find sponsor logos in the most exciting frames.
""")
            
    st.subheader("Geospatial analysis of Fan Engagement")
    with st.container():
        with st.expander("We analyzed thousands of IPL tweets and maps where cricket fans are most active. Based on this data, it created small geographic zones and gave sponsor-wise suggestions for delivery apps, local businesses, and service providers. The goal was to help brands target areas with high fan activity more effectively.", expanded=False):
            st.markdown("""
Mapped All Locations
The tweets were matched with their coordinates, city, neighborhood, and more location info.

Grouped Fans into Zones
Created small zones of fans living near each other (within 2 km). These were called delivery zones.

Analyzed Each Zone
For every zone, it calculated:

Number of fans

Main city/area

Postal codes and neighborhoods

Targeting precision: HIGH (postal code), MEDIUM (neighborhood), LOW (only area info)

Generated Sponsor Insights
Suggested which zones were best for:

Food delivery apps (needs high fan count + postal codes)

Local shops & cafes (needs moderate fan count)

Event promotions (all zones useful)

Created an Interactive Map
Made a colorful map showing all zones, with different sizes and colors based on fan count and precision level.
""")

# =========================================================
# End of script
# =========================================================
