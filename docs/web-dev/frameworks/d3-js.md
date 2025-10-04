# D3.js Framework

![D3.js Logo](../../assets/images/web-dev/d3-js-logo.png){.center}

> D3.js (Data-Driven Documents) is a powerful JavaScript library for creating dynamic, interactive data visualizations in web browsers. <br />
> It leverages web standards such as SVG, HTML, and CSS to bring data to life.
> The main advantage of D3.js is its ability to bind data to the Document Object Model (DOM) and apply data-driven transformations to the document.

## Basic Concepts

### Selections

D3.js uses selections to select and manipulate DOM elements. You can select elements using CSS selectors.

```javascript
// Select all paragraph elements
d3.selectAll("p")
```

> The result is an array of selected elements. So you can use array methods like `forEach`, `map`, etc., on the selection.

You can also select a single element:

```javascript
// Select the first paragraph element
d3.select("p")
```

### Attributes and Styles

You can set attributes and styles of selected elements using the `attr` and `style` methods.

```javascript
// Set the class attribute of all paragraph elements
d3.selectAll("p")
    .attr("class", "my-class")

// Set the color style of all paragraph elements
d3.selectAll("p")
    .style("color", "blue")
```

### Data Binding

One of the most powerful features of D3.js is data binding. You can bind data to DOM elements and create new elements based on the data.

```javascript
// Sample data
const myData = [
    {x: 30, y: 20},
    {x: 80, y: 90},
    {x: 130, y: 50}
];

// Bind myData to circle elements
d3.selectAll("circle")
    .data(myData)
    .enter().append("circle") // Create a circle for each data point
    .attr("cx", d => d.x)
    .attr("cy", d => d.y)
    .attr("r", 10)
    .style("fill", "red");
```

#### Enter, Update, and Exit

When binding data to elements, D3.js provides three key methods to manage the lifecycle of elements: `enter()`, `update()`, and `exit()`.

- `enter()`: Creates new elements for any data points that do not have corresponding DOM elements.
- `update()`: Updates existing elements that correspond to data points.
- `exit()`: Removes elements that no longer have corresponding data points.

```javascript title="simple-example,js"
const myData = [10, 20, 30];

d3.selectAll("circle")
    .data(myData)
    .join(
        enter => enter.append("circle")         // Create new circles for new data points
                      .attr("r", d => d)
                      .attr("cx", (d, i) => i * 50 + 25)
                      .attr("cy", 50)
                      .style("fill", "green"),
        update => update.style("fill", "blue"), // when data is updated, existing circles turn blue
        exit => exit.remove()                   // when data is removed, remove the corresponding circles
    );
```

> You mostly use this together with transitions to create smooth animations when data changes.

### Scales

Scales are functions that map input data to output values, such as pixel positions or colors. <br />
D3.js provides several types of scales, including linear, ordinal, and time scales.

```javascript
// Create a linear scale
const xScale = d3.scaleLinear()
    .domain([0, 1])     // Input domain
    .range([-10, 10]);  // Output range

xScale(0.5); // Returns 0
xScale(0);   // Returns -10
```

> range can also be other things like:
> 
> - colors: `.range(["red", "blue"])` -> maps hex values
> - ...

### Axes

D3.js provides a convenient way to create axes for your visualizations using the `d3.axis` module.

```javascript
// Create a linear scale for the x-axis
const xScale = d3.scaleLinear()
    .domain([0, 100])
    .range([0, 400]);

// Create the x-axis
const xAxis = d3.axisBottom(xScale)
    .ticks(5)                       // _optional_ number of ticks
    .tickFormat(d3.format(".0f"));  // _optional_ format ticks as integers

// Append the x-axis to an SVG element
d3.select("svg")
    .append("g")
    .attr("class", "x-axis")
    .call(xAxis); // Call the axis generator
```

> You can cutsomize the axis further via CSS.

### Transitions

Transitions allow you to animate changes to your visualizations over time.

```javascript
// Select all circles and transition their radius to 20 over 1 second
d3.selectAll("circle")
    .transition()
    .duration(1000)         // Duration in milliseconds
    .attr("r", 20);         // New radius
    .ease(d3.easeBounce);   // _optional_ easing function
```

> You can chain multiple transitions together for more complex animations.

### Interactions

D3.js supports user interactions such as mouse events and touch events.

```javascript
// Add a click event listener to all circles
// when a circle is clicked, its color changes to orange
d3.selectAll("circle")
    .on("click", function(event, d) {
        d3.select(this)                 // 'this' refers to the clicked circle
            .style("fill", "orange");   // Change color to orange on click
    });
```

There are several types of events you can listen to, including:

- `click`: when an element is clicked
- `mouseover`: when the mouse pointer is over an element
- `mouseout`: when the mouse pointer leaves an element
- ...

## Types of Visualizations

D3.js has a wide range of predefined layouts and shapes that you can use to create various types of visualizations, including:

- Bar charts: `d3.bar()`
- Line charts: `d3.line()`
- Pie charts: `d3.pie()`
- Scatter plots: `d3.scatter()`
- Force-directed graphs: `d3.forceSimulation()`
- Hierarchical layouts: `d3.hierarchy()`, `d3.tree()`, `d3.cluster()`
- Maps: `d3.geoPath()`, `d3.geoProjection()`
- ...

