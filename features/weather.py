
import streamlit as st


def display_weather_gui(data):
    """Renders a modern, interactive Weather GUI."""
    with st.container(border=True):
        # Header Area
        col_city, col_time = st.columns([2, 1])
        with col_city:
            st.subheader(f"📍 {data.get('city', 'Hyderabad, Sindh')}")
        with col_time:
            st.caption(f"📅 {data.get('date', 'Feb 23, 2026')}")

        # Main Temperature Display
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown(f"<h1 style='text-align: center; font-size: 60px;'>{data.get('temp')}°C</h1>", unsafe_allow_html=True)
        with c2:
            st.write("") # Spacer
            st.write(f"**{data.get('condition', 'Mainly Clear')}**")
            st.write(f"Feels like: {data.get('feels_like')}°C")
        with c3:
            # Simple icon mapping logic
            icon = "☀️" if "clear" in data.get('condition', '').lower() else "☁️"
            st.markdown(f"<h1 style='text-align: center; font-size: 60px;'>{icon}</h1>", unsafe_allow_html=True)

        st.divider()

        # Environmental Stats (Scrollable/Horizontal)
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Humidity", f"{data.get('humidity')}%")
        m2.metric("Wind", f"{data.get('wind')} km/h")
        m3.metric("Visibility", f"{data.get('visibility')} km")
        m4.metric("UV Index", "Low")

        # Interactive Forecast (Scrollable Container)
        st.write("### 7-Day Forecast")
        with st.container(height=150): # This makes it scrollable
            f_cols = st.columns(7)
            days = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
            for i, col in enumerate(f_cols):
                col.write(days[i])
                col.write("☀️")
                col.caption(f"{24 - i}°")