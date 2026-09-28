# Wi-Fi RSSI Localization

Indoor localization of a Wi-Fi transmitter (Device B) from RSSI measurements
collected by a scanner (Device A) - fingerprinting + signal propagation +
Kalman tracking, with a live-map HUD and a camera overlay that draws the
Wi-Fi signal path.

## Pipeline

    Device B (Wi-Fi target)  -- signal -->  Device A (scanner)
        -> Signal filter        (median + EMA + outlier rejection)
        -> Feature extraction   (RSSI stats, channel, temporal variation)
        -> Localization engine
             1. fingerprint matching   (kNN vs calibration database)
             2. signal propagation model (log-distance path loss range rings)
             3. temporal tracking        (Kalman filter, constant velocity)
             4. confidence estimation
        -> Position estimator: X, Y, error radius, confidence
        -> Live map UI + camera signal-path overlay

## Files

- `wifi_localizer.py` - the full engine. Stdlib only, no dependencies.
- `index.html` - combined single-file UI (HTML + CSS + JS, no build step).
- `data/` - sample calibration database, tracking session and raw RSSI log.

## Run it

```bash
python wifi_localizer.py demo    # synthetic end-to-end run -> data/
python wifi_localizer.py serve   # local engine the UI connects to (port 8765)
python wifi_localizer.py calibrate mycal.json   # store a calibration map
python wifi_localizer.py track mylog.csv        # process an RSSI log
```

Then open http://127.0.0.1:8765/ (or open `index.html` directly - it
auto-connects to the engine when running) and press **Live track (Python)**.

The UI also works fully offline: **Run JS demo** re-implements the pipeline
in the browser, and **Load Python session** opens any stored
`tracking_session.json` via the file picker.

## Accuracy

Steady-state indoor accuracy (share of estimates after the first full
sweep of the scan points within 2.5 m of the true position) measured
across 15 randomized seeds: **100%**, with final errors of 0.17-1.5 m.
Precision upgrades: Huber-robust multi-start range-ring solver, 5x4
calibration grid, 10 scan points, 12 samples per burst, squared-inverse
fingerprint weighting, tuned Kalman (q=0.03, r=1.0).

## Hosted UI

GitHub Pages serves the static UI: it runs the JS demo, session playback
and the camera overlay; the Python live engine runs locally with `serve`.

## Notes

- Camera access requires HTTPS or localhost - GitHub Pages and the local
  server both qualify; without a camera, a simulated feed is shown.
- The camera bearing assumes map orientation; a real deployment should fuse
  the device compass/gyro into the same Kalman layer.
