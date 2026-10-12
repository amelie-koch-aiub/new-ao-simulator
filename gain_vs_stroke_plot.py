import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv('ao_performance.csv')

stroke05 = df[(df['simulation_ID'] >= 37) & (df['simulation_ID'] <= 40)]
stroke1 = df[(df['simulation_ID'] >= 41) & (df['simulation_ID'] <= 44)]
stroke2 = df[(df['simulation_ID'] >= 45) & (df['simulation_ID'] <= 48)]
stroke4 = df[(df['simulation_ID'] >= 49) & (df['simulation_ID'] <= 52)]
stroke6 = df[(df['simulation_ID'] >= 53) & (df['simulation_ID'] <= 56)]

plt.plot(stroke05['gain'], stroke05['avg_Strehl_AO']/stroke05['avg_Strehl_noAO'], label='stroke 0.5µm')
plt.plot(stroke1['gain'], stroke1['avg_Strehl_AO']/stroke1['avg_Strehl_noAO'], label='stroke 1µm')
plt.plot(stroke2['gain'], stroke2['avg_Strehl_AO']/stroke2['avg_Strehl_noAO'], label='stroke 2µm')
plt.plot(stroke4['gain'], stroke4['avg_Strehl_AO']/stroke4['avg_Strehl_noAO'], label='stroke 4µm')
plt.plot(stroke6['gain'], stroke6['avg_Strehl_AO']/stroke6['avg_Strehl_noAO'], label='stroke 6µm')
plt.legend()
plt.xlabel('gain')
plt.ylabel('Strehl corr/Strehl uncorr (avg. over 200 iter.)')
plt.title('Performance increase for different DM strokes and gains')
plt.savefig('gain_vs_stroke_test_avgStrehl.png')
plt.clf()

plt.plot(stroke05['gain'], stroke05['peak_Strehl_AO']/stroke05['avg_Strehl_noAO'], label='stroke 0.5µm')
plt.plot(stroke1['gain'], stroke1['peak_Strehl_AO']/stroke1['avg_Strehl_noAO'], label='stroke 1µm')
plt.plot(stroke2['gain'], stroke2['peak_Strehl_AO']/stroke2['avg_Strehl_noAO'], label='stroke 2µm')
plt.plot(stroke4['gain'], stroke4['peak_Strehl_AO']/stroke4['avg_Strehl_noAO'], label='stroke 4µm')
plt.plot(stroke6['gain'], stroke6['peak_Strehl_AO']/stroke6['avg_Strehl_noAO'], label='stroke 6µm')
plt.legend()
plt.xlabel('gain')
plt.ylabel('Strehl corr/Strehl uncorr (max. over 200 iter.)')
plt.title('Performance increase for different DM strokes and gains')
plt.savefig('gain_vs_stroke_test_peakStrehl.png')