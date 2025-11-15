---
title: "CM3604 – Deep Learning Coursework Element 2"
output: html_document
---

# CM3604 – Deep Learning  
## Coursework Element 2 – Group Project

### Module Information
- **Module:** CM3604 – Deep Learning  
- **Semester:** 1 (Academic Year 2025/2026)  
- **Assessment Type:** Coursework Element 2 (Group Work)  
- **Group Size:** 3–4 members (one submission per group)  
- **Deadline:** **16th November 2025 – 23:59**  
- **Submission:** Upload to Assessment Dropbox in CampusMoodle  
- **Word Limit:** N/A  
- **Module Coordinator:** Prasan Yapa  

---

# Purpose of the Assessment
This project provides hands-on experience developing Transformer-based neural models. You will implement a simplified Transformer encoder and later extend it into a full Transformer language model.

### Learning Outcomes
1. Build neural models using appropriate functions and parameters in a modern Python framework.  
2. Implement, test, and apply deep learning architectures in data science applications.

---

# Part 1: Transformer Encoder (50 Marks)

## Objective
Build a simplified Transformer encoder for a 3-class sequence classification task.  
Given a 20-character string, predict at each position how many times that character has appeared **earlier** (labels: 0, 1, or 2+).

A second variant counts occurrences both **before and after**, run with:

