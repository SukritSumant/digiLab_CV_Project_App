import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.cm as cm
import os

# ----- Setup -----
st.set_page_config(layout="wide")
st.title("Plasma EFIT Shape Parameter Prediction Visualiser")

# ----- Load Data -----
st.sidebar.header("Shot Selection")

@st.cache_data
def load_metadata():
	return pd.read_csv("elongation_predictions.csv")

df = load_metadata()

# ----- Scatter Plot -----
st.subheader("Model Validation Plot")

fig = px.scatter(
	df,
	x="elongation_true",
	y="elongation_pred",
	error_y=2 * df["elongation_std"],
	hover_data=["shots"],
	color=(abs(df["elongation_true"] - df["elongation_pred"]) > 2 * df["elongation_std"]).astype(str),
	labels={"color": "Outlier"},
	title="True vs Predicted Elongation (with 2\u03c3 bounds)"
)

min_val = min(df["elongation_true"].min(), df["elongation_pred"].min())
max_val = max(df["elongation_true"].max(), df["elongation_pred"].max())

fig.add_trace(go.Scatter(
	x=[min_val, max_val],
	y=[min_val, max_val],
	mode="lines",
	line=dict(dash="dash", color="gray"),
	name="Ideal: y = x"
))
fig.update_traces(marker=dict(size=8, line=dict(width=1, color='DarkSlateGrey')))
fig.update_layout(height=600)

st.plotly_chart(fig, use_container_width=True)

# ----- Shot Selection -----
selected_shot = st.sidebar.selectbox("Select a shot to inspect", df["shots"].unique())
st.sidebar.write("Frame: 50% (fixed), Camera: B")

# Get metadata row
row = df[df["shots"] == selected_shot].iloc[0]
st.markdown(f"### Shot {selected_shot}")
st.write(f"**EFIT Elongation**: {row.elongation_true:.3f}")
st.write(f"**Predicted Elongation**: {row.elongation_pred:.3f} ± {2*row.elongation_std:.3f} (2σ)")

# ----- Load .npz Data -----
@st.cache_data
def load_npz_data(shot):
	file_path = f"processed_shots/{shot}.npz"
	if not os.path.exists(file_path):
		return None
	data = np.load(file_path)
	return data["frame"], data["mask"], data["edges"], data["coeffs"]

result = load_npz_data(selected_shot)

if result is None:
	st.error(f"No processed data found for shot {selected_shot}.")
else:
	frame, mask, edges, coeffs = result
	col1, col2 = st.columns(2)

	with col1:
		st.markdown("<div style='padding-left: 200px;'><b>Raw Frame</b></div>", unsafe_allow_html=True)
		fig1, ax1 = plt.subplots(figsize=(4.5, 4.5))
		norm_frame = frame.astype(np.float32) / 255.0
		plasma_frame = cm.get_cmap('plasma')(norm_frame)[:, :, :3]
		ax1.imshow(plasma_frame)
		ax1.axis("off")
		st.pyplot(fig1)

	with col2:
		st.markdown("<div style='padding-left: 170px;'><b>Segmentation Mask Overlay</b></div>", unsafe_allow_html=True)
		fig2, ax2 = plt.subplots(figsize=(4.5, 4.5))
		ax2.imshow(np.stack([frame]*3, axis=-1))  # Convert grayscale to RGB
		ax2.imshow(mask, alpha=0.5, cmap='jet')
		ax2.axis("off")
		st.pyplot(fig2)

	# ----- Polynomial Fit over Edge Map -----
	# Fit plot below the two columns, centred in the layout
	# Add vertical space and centred heading

	# st.markdown(" ")
	# st.markdown("<div style='text-align: center;'><b>Polynomial Fit on Plasma Edge</b></div>", unsafe_allow_html=True)

	# # Use three columns, place content in the centre one
	# col_left, col_mid, col_right = st.columns([1, 2, 1])  # Adjust 1:2:1 as needed

	# with col_mid:
	# 	fig3, ax3 = plt.subplots(figsize=(5, 5))  # Same size as above
	# 	ys, xs = np.nonzero(edges)
	# 	ax3.imshow(mask, cmap="gray")
	# 	ax3.scatter(xs, ys, s=1, color="gold", label="Edge Pixels")

	# 	if len(xs) >= 3:
	# 		poly = np.poly1d(coeffs)
	# 		y_fit = np.linspace(ys.min(), ys.max(), 300)
	# 		x_fit = poly(y_fit)
	# 		ax3.plot(x_fit, y_fit, color="red", linewidth=2, label="Polynomial Fit")

	# 	ax3.axis("equal")
	# 	ax3.axis("off")
	# 	ax3.legend(loc = "upper center")
	# 	st.pyplot(fig3)
	# # ---- Polynomial Coefficient Display ----
	# st.markdown(" ")
	# st.markdown("<div style='text-align: center;'><b>Fitted Polynomial Coefficients</b></div>", unsafe_allow_html=True)

	# terms = [f"x^{i}" for i in reversed(range(len(coeffs)))]
	# formatted = [
	# 	f"<div style='text-align: center;'>{term} term: {coeff:.4f}</div>"
	# 	for term, coeff in zip(terms, coeffs)
	# ]
	# for line in formatted:
	# 	st.markdown(line, unsafe_allow_html=True)

	# Add vertical space and heading
st.markdown(" ")
st.markdown("<div style='padding-left: 170px;'<b>Polynomial Fit on Plasma Edge</b></div>", unsafe_allow_html=True)

# Side-by-side: fit on the left, LaTeX coefficients on the right
col_fit, col_coeff = st.columns([1, 1])  # Wider column for the figure

with col_fit:
	fig3, ax3 = plt.subplots(figsize=(4.5, 4.5))
	ys, xs = np.nonzero(edges)
	ax3.imshow(mask, cmap="gray")
	ax3.scatter(xs, ys, s=1, color="gold", label="Edge Pixels")

	if len(xs) >= 3:
		poly = np.poly1d(coeffs)
		y_fit = np.linspace(ys.min(), ys.max(), 300)
		x_fit = poly(y_fit)
		ax3.plot(x_fit, y_fit, color="red", linewidth=2, label="Polynomial Fit")

	ax3.axis("equal")
	ax3.axis("off")
	ax3.legend(loc="upper center")
	st.pyplot(fig3)

with col_coeff:
	st.markdown(" ")
	st.markdown(" ")
	st.markdown(" ")
	st.markdown(" ")
	st.markdown(" ")
	st.markdown(" ")
	st.markdown(" ")
	st.markdown(" ")
	st.markdown(" ")

	st.markdown("<div style='padding-left: 170px;'><b>Fitted Polynomial Coefficients</b></div>", unsafe_allow_html=True)
	powers = list(reversed(range(len(coeffs))))
	powers = list(reversed(range(len(coeffs))))
	for i, coeff in zip(powers, coeffs):
		if coeff< 0:
			st.latex(rf"\text{{x}}^{i} \text{{ term: }} {coeff:.4f}")
		else:
			st.latex(rf"\text{{x}}^{i} \text{{ term: }} +{coeff:.4f}")



