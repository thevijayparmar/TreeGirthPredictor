# 🌳 Tree Allometry ML – FULL Colab mini-app (with ROTATABLE Plotly 3-D)
# --------------------------------------------------------------------
# 1️⃣  Install deps + imports
!pip install -q ipywidgets scikit-learn matplotlib plotly==5.*

import pandas as pd, numpy as np, matplotlib.pyplot as plt, warnings, os, plotly.express as px
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
import ipywidgets as w
from IPython.display import display, Markdown
from google.colab import files
warnings.filterwarnings("ignore", message="X does not have valid feature names")

def md(txt): display(Markdown(txt))

# --------------------------------------------------------------------
# 2️⃣  Expected format preview
md("## 📑 **CSV must have**&nbsp;`Tree Name, Girth (m), Height (m)`")
display(pd.DataFrame({"Tree Name":["ExampleTree"],
                      "Girth (m)":[0.45],
                      "Height (m)":[12.3]}))

# --------------------------------------------------------------------
# 3️⃣  Upload training data & fit models
md("## 1️⃣ **Upload training CSV**")
up = files.upload()                # drag-drop the file
csv_path = next(iter(up))
df = pd.read_csv(csv_path, na_values=["NIL",""])
df["Girth (m)"]  = pd.to_numeric(df["Girth (m)"],  errors="coerce")
df["Height (m)"] = pd.to_numeric(df["Height (m)"], errors="coerce")
drop = df.isna().any(axis=1).sum()
df = df.dropna(subset=["Tree Name","Girth (m)","Height (m)"])
md(f"✅ Using **{len(df)} rows**  (dropped {drop} bad rows)")

enc   = OneHotEncoder(sparse_output=False).fit(df[["Tree Name"]])
cats  = enc.transform(df[["Tree Name"]])
X_g2h = np.hstack([df[["Girth (m)"]].values,  cats]);  y_h = df["Height (m)"].values
X_h2g = np.hstack([df[["Height (m)"]].values, cats]);  y_g = df["Girth (m)"].values
rf_g2h = RandomForestRegressor(n_estimators=200, random_state=0).fit(X_g2h, y_h)
rf_h2g = RandomForestRegressor(n_estimators=200, random_state=0).fit(X_h2g, y_g)

species     = df["Tree Name"].unique()
train_cnt   = df["Tree Name"].value_counts().to_dict()
overall_r2  = rf_g2h.score(X_g2h, y_h)
sp_r2       = {sp:(rf_g2h.score(X_g2h[df["Tree Name"]==sp], y_h[df["Tree Name"]==sp])
                   if (df["Tree Name"]==sp).sum()>2 else overall_r2)
               for sp in species}

# --------------------------------------------------------------------
# 4️⃣  Training summary + global 2-D scatter
md("## 2️⃣ **Training summary**")
display(pd.DataFrame({"Tree":list(train_cnt),"Samples":list(train_cnt.values())}))
md(f"- Overall R² (Girth→Height): **{overall_r2:.2f}**")

plt.figure(figsize=(5,4))
plt.scatter(df["Girth (m)"], df["Height (m)"], alpha=.6)
plt.xlabel("Girth (m)"); plt.ylabel("Height (m)")
plt.title("All species • Height vs Girth")
plt.grid(True); plt.show()

# utility for prediction spread
def tree_std(model, feat):
    return float(np.std([t.predict(feat)[0] for t in model.estimators_]))

# --------------------------------------------------------------------
# 5️⃣  Interactive UI
container    = w.VBox()
mode_toggle  = w.ToggleButtons(options=["Single tree","Multiple trees"], description="Mode:")
reset_btn    = w.Button(description="🔄 Reset UI", button_style="warning")
display(Markdown("## 3️⃣ **Prediction mode**"))
display(w.HBox([reset_btn, mode_toggle]), container)

# ----- SINGLE tree panel -----
def make_single_ui():
    head = w.HTML("<h4>➡️ Single-tree prediction</h4>")
    sp   = w.Dropdown(options=species, description="Tree:")
    gir  = w.FloatText(description="Girth (m):", value=np.nan)
    hgt  = w.FloatText(description="Height (m):", value=np.nan)
    btn  = w.Button(description="Predict", button_style="success")
    out  = w.Output(); plot2d = w.Output(); plot3d = w.Output()

    def on_click(_):
        out.clear_output(); plot2d.clear_output(); plot3d.clear_output()
        vec = enc.transform([[sp.value]]).flatten()
        conf_sp = sp_r2.get(sp.value, overall_r2)*100

        if not np.isnan(gir.value) and np.isnan(hgt.value):  # known girth
            feat = np.array([[gir.value, *vec]]); pred = rf_g2h.predict(feat)[0]
            spread = tree_std(rf_g2h, feat)
            x_val, y_val = gir.value, pred
            with out: md(f"🔸 Pred **Height = {pred:.2f} m**  (Conf≈{conf_sp:.1f} %, ±{spread:.2f})")
        elif not np.isnan(hgt.value) and np.isnan(gir.value):  # known height
            feat = np.array([[hgt.value, *vec]]); pred = rf_h2g.predict(feat)[0]
            spread = tree_std(rf_h2g, feat)
            x_val, y_val = pred, hgt.value
            with out: md(f"🔸 Pred **Girth = {pred:.2f} m**  (Conf≈{conf_sp:.1f} %, ±{spread:.2f})")
        else:
            with out: md("⚠️ Enter **exactly one** of Girth or Height")
            return

        sub = df[df["Tree Name"]==sp.value]

        # 2-D scatter
        with plot2d:
            plt.figure(figsize=(5,4))
            plt.scatter(sub["Girth (m)"], sub["Height (m)"], alpha=.7, label="training")
            plt.scatter([x_val],[y_val],c="red",marker="D",s=80,label="your point")
            plt.xlabel("Girth (m)"); plt.ylabel("Height (m)")
            plt.title(f"{sp.value}: Height vs Girth (2-D)")
            plt.legend(); plt.grid(True); plt.show()

        # 3-D scatter – Plotly (drag to rotate)
        with plot3d:
            fig = px.scatter_3d(
                sub, x="Girth (m)", y="Height (m)", z="Height (m)",
                opacity=0.7, color_discrete_sequence=["steelblue"],
                labels={"Girth (m)":"Girth (m)", "Height (m)":"Height (m)"},
                title=f"{sp.value}: 3-D view (rotate me)"
            )
            fig.add_scatter3d(
                x=[x_val], y=[y_val], z=[y_val],
                mode="markers",
                marker=dict(size=6, color="red", symbol="diamond"),
                name="your point"
            )
            fig.update_layout(legend=dict(x=0.02, y=0.97))
            fig.show()

    btn.on_click(on_click)
    return w.VBox([head, sp, gir, hgt, btn, out, plot2d, plot3d])

# ----- MULTIPLE tree panel -----
def make_multi_ui():
    head = w.HTML("<h4>➡️ Multiple-tree prediction</h4>")
    note = w.HTML("Upload CSV (same columns). Leave one of Girth/Height blank.")
    up   = w.FileUpload(accept=".csv", multiple=False)
    out  = w.Output()

    def do_upload(change):
        if not up.value: return
        (fname, meta), = up.value.items()
        open(fname,"wb").write(meta["content"])
        df2 = pd.read_csv(fname, na_values=["NIL",""])
        df2["Girth (m)"]  = pd.to_numeric(df2["Girth (m)"],  errors="coerce")
        df2["Height (m)"] = pd.to_numeric(df2["Height (m)"], errors="coerce")

        preds, confs, stds, tcount = [],[],[],[]
        for i,r in df2.iterrows():
            sp = r["Tree Name"]; vec = enc.transform([[sp]]).flatten()
            confs.append(sp_r2.get(sp, overall_r2)*100)
            tcount.append(train_cnt.get(sp,0))
            if pd.isna(r["Height (m)"]) and not pd.isna(r["Girth (m)"]):
                feat = np.array([[r["Girth (m)"],*vec]]); p=rf_g2h.predict(feat)[0]
                df2.at[i,"Height (m)"]=p; preds.append(p); stds.append(tree_std(rf_g2h,feat))
            elif pd.isna(r["Girth (m)"]) and not pd.isna(r["Height (m)"]):
                feat = np.array([[r["Height (m)"],*vec]]); p=rf_h2g.predict(feat)[0]
                df2.at[i,"Girth (m)"]=p; preds.append(p); stds.append(tree_std(rf_h2g,feat))
            else:
                preds.append(np.nan); stds.append(np.nan)

        df2["Predicted"]=preds; df2["Confidence (%)"]=confs
        df2["Tree-Std"]=stds;   df2["Train Count"]=tcount
        out_csv="predictions.csv"; df2.to_csv(out_csv,index=False)

        out.clear_output()
        with out:
            display(df2.head())
            md("⬇️ **Download full CSV**")
            files.download(out_csv)

    up.observe(do_upload, names="value")
    return w.VBox([head, note, up, out])

# ----- switch & reset logic
def render(*_):
    container.children = (make_single_ui(),) if mode_toggle.value=="Single tree" else (make_multi_ui(),)
def reset(_): render()
mode_toggle.observe(render, names="value")
reset_btn.on_click(reset)
render()          # first draw
