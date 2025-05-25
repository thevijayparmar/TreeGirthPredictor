"""
Streamlit Tree Allometry ML demo
Upload a training CSV  (Tree Name, Girth (m), Height (m))
Then:
  • single-tree interactive prediction (rotatable 3-D Plotly scatter)
  • bulk CSV fill + download
"""

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
import plotly.express as px
from io import BytesIO

st.set_page_config(page_title="Tree Allometry ML", layout="wide")

# ──────────────────── helper
@st.cache_data(show_spinner=True)
def train_models(df: pd.DataFrame, n_estimators=200):
    enc = OneHotEncoder(sparse_output=False).fit(df[["Tree Name"]])
    cats = enc.transform(df[["Tree Name"]])

    X_g2h = np.hstack([df[["Girth (m)"]].values,  cats])
    X_h2g = np.hstack([df[["Height (m)"]].values, cats])
    y_h   = df["Height (m)"].values
    y_g   = df["Girth (m)"].values

    g2h = RandomForestRegressor(n_estimators=n_estimators).fit(X_g2h, y_h)
    h2g = RandomForestRegressor(n_estimators=n_estimators).fit(X_h2g, y_g)

    overall_r2 = g2h.score(X_g2h, y_h)
    sp_r2 = {}
    for sp in df["Tree Name"].unique():
        msk = df["Tree Name"] == sp
        sp_r2[sp] = g2h.score(X_g2h[msk], y_h[msk]) if msk.sum() > 2 else overall_r2

    return enc, g2h, h2g, overall_r2, sp_r2

def tree_std(model, feat):
    return float(np.std([t.predict(feat)[0] for t in model.estimators_]))

# ──────────────────── sidebar - Training upload
st.sidebar.header("1. Upload *training* CSV")
train_csv = st.sidebar.file_uploader(
    "Columns : Tree Name, Girth (m), Height (m)", type="csv", key="train"
)

if train_csv:
    df_train = pd.read_csv(train_csv, na_values=["NIL", ""])
    df_train["Girth (m)"]  = pd.to_numeric(df_train["Girth (m)"],  errors="coerce")
    df_train["Height (m)"] = pd.to_numeric(df_train["Height (m)"], errors="coerce")
    df_train.dropna(subset=["Tree Name","Girth (m)","Height (m)"], inplace=True)

    st.success(f"Training rows : {len(df_train)} | Species : {df_train['Tree Name'].nunique()}")
    enc, g2h, h2g, overall_r2, sp_r2 = train_models(df_train)

    # global scatter
    st.subheader("Training scatter (all species)")
    fig_glob = px.scatter(
        df_train, x="Girth (m)", y="Height (m)",
        opacity=0.6, title="Height vs Girth (all)")
    st.plotly_chart(fig_glob, use_container_width=True)

    tab_single, tab_bulk = st.tabs(["🌳 Single tree", "📂 Multiple trees"])

    # ─────────── Single-tree tab
    with tab_single:
        st.header("Interactive single-tree prediction")
        species = sorted(df_train["Tree Name"].unique())
        sel_sp  = st.selectbox("Species", species)
        col1, col2 = st.columns(2)
        inp_g   = col1.number_input("Known Girth (m) (leave blank if unknown)",
                                    min_value=0.0, step=0.01, format="%.2f", key="g")
        inp_h   = col2.number_input("Known Height (m) (leave blank if unknown)",
                                    min_value=0.0, step=0.1, format="%.2f", key="h")
        predict_btn = st.button("Predict")

        if predict_btn:
            vec = enc.transform([[sel_sp]]).flatten()
            conf_sp = sp_r2.get(sel_sp, overall_r2)*100
            if (inp_g and not inp_h) or (inp_h and not inp_g):
                if inp_g and not inp_h:
                    feat = np.array([[inp_g, *vec]])
                    pred = g2h.predict(feat)[0]
                    spread = tree_std(g2h, feat)
                    msg   = f"**Predicted Height = {pred:.2f} m**  (Conf≈{conf_sp:.1f}% ±{spread:.2f})"
                    x_val, y_val = inp_g, pred
                else:
                    feat = np.array([[inp_h, *vec]])
                    pred = h2g.predict(feat)[0]
                    spread = tree_std(h2g, feat)
                    msg   = f"**Predicted Girth = {pred:.2f} m**  (Conf≈{conf_sp:.1f}% ±{spread:.2f})"
                    x_val, y_val = pred, inp_h
                st.markdown(msg)

                sub = df_train[df_train["Tree Name"] == sel_sp]
                fig2d = px.scatter(sub, x="Girth (m)", y="Height (m)",
                                   opacity=0.7, title=f"{sel_sp} • 2-D scatter")
                fig2d.add_scatter(x=[x_val], y=[y_val],
                                  mode="markers", marker=dict(color="red", size=10,
                                  symbol="diamond"), name="your point")
                st.plotly_chart(fig2d, use_container_width=True)

                fig3d = px.scatter_3d(
                    sub, x="Girth (m)", y="Height (m)", z="Height (m)",
                    opacity=0.7, title=f"{sel_sp} • 3-D scatter (drag to rotate)")
                fig3d.add_scatter3d(
                    x=[x_val], y=[y_val], z=[y_val],
                    mode="markers", marker=dict(color="red", size=5, symbol="diamond"),
                    name="your point",
                )
                st.plotly_chart(fig3d, use_container_width=True)

            else:
                st.error("Enter **exactly one** of Girth or Height.")

    # ─────────── Bulk-CSV tab
    with tab_bulk:
        st.header("Fill an entire CSV")
        st.markdown(
            "*Upload a second CSV with the same columns. Leave exactly one of "
            "`Girth (m)` / `Height (m)` blank in each row.*"
        )
        pred_csv = st.file_uploader("Prediction CSV", type="csv", key="pred")
        if pred_csv:
            df_pred = pd.read_csv(pred_csv, na_values=["NIL",""])
            df_pred["Girth (m)"]  = pd.to_numeric(df_pred["Girth (m)"],  errors="coerce")
            df_pred["Height (m)"] = pd.to_numeric(df_pred["Height (m)"], errors="coerce")

            preds, confs, stds, tcnt = [],[],[],[]
            for i,r in df_pred.iterrows():
                sp = r["Tree Name"]
                vec = enc.transform([[sp]]).flatten()
                confs.append(sp_r2.get(sp, overall_r2)*100)
                tcnt.append(train_cnt:=df_train["Tree Name"].value_counts().get(sp,0))

                if pd.isna(r["Height (m)"]) and not pd.isna(r["Girth (m)"]):
                    feat = np.array([[r["Girth (m)"],*vec]])
                    p = g2h.predict(feat)[0]; df_pred.at[i,"Height (m)"]=p
                    preds.append(p); stds.append(tree_std(g2h,feat))
                elif pd.isna(r["Girth (m)"]) and not pd.isna(r["Height (m)"]):
                    feat = np.array([[r["Height (m)"],*vec]])
                    p = h2g.predict(feat)[0]; df_pred.at[i,"Girth (m)"]=p
                    preds.append(p); stds.append(tree_std(h2g,feat))
                else:
                    preds.append(np.nan); stds.append(np.nan)

            df_pred["Predicted"] = preds
            df_pred["Confidence (%)"] = confs
            df_pred["Tree-Std"] = stds
            df_pred["Train Count"] = tcnt

            st.success("✅ Predictions complete")
            st.dataframe(df_pred.head())

            # download-link
            csv_buf = BytesIO()
            df_pred.to_csv(csv_buf, index=False)
            st.download_button("⬇️ Download full CSV", data=csv_buf.getvalue(),
                               file_name="predictions.csv", mime="text/csv")
else:
    st.info("↑ Upload a training CSV in the sidebar to begin.")
