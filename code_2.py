input_csv_path = 'file_path' # Путь к файлу с датасетом.
target_text_column = 'column_name' # Название столбца для оценки.
labels_to_check = ['приказной_тон', 'отказ_помочь', 'наезды_оскорбления', 'канцелярит']

data_to_process = pd.read_csv(input_csv_path)
print(f"--- Анализ {len(data_to_process)} элементов… ---")

processed_results = evaluate_csv(data_to_process, model, tokenizer, labels_to_check, target_text_column)

print("\n--- Обзор результатов ---")
counts = {label: (processed_results[label] > 0.5).sum() for label in labels_to_check}
for label, count in counts.items():
    percentage = (count / len(data_to_process)) * 100
    print(f"«{label}» — {count} ({percentage:.1f}%)")

avg_empathy = processed_results['HUMAN_LIKENESS_SCORE'].mean()
print(f"\nОбщая оценка эмпатийности — {avg_empathy:.2f}/1.00.")

output_filename = 'final_output.csv'
processed_results.to_csv(output_filename, index=False)
print(f"\nОбработанный файл сохранён как «{output_filename}».")