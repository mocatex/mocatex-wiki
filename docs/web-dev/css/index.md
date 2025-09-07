# CSS - The Basics

> CSS (Cascading Style Sheets) is a stylesheet language used to describe the presentation of a document written in HTML or XML. It controls the layout, colors, fonts, and overall visual appearance of web pages.

## How to inlcude CSS in HTML

There are three main ways to include CSS in an HTML document:

1. :x: **Inline CSS**: Using the `style` attribute within HTML elements. -> Very bad

   ```html
   <p style="color: blue; font-size: 16px;">This is a blue paragraph.</p>
   ```

2. :warning: **Internal CSS**: Using a `<style>` tag within the `<head>` section of the HTML document. -> also bad

   ```html
   <head>
       <style>
           h1 {
               color: navy;
           }
       </style>
   </head>
   ```

3. :white_check_mark: **External CSS**: Linking to external CSS file(s) using the `<link>` tag within the `<head>` section. -> Best practice

   ```html
    <head>
         <link rel="stylesheet" href="styles.css">
    </head>
    ```

Then in those files you can write your CSS code!

