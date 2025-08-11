# How to write this documentation?

This documentation is written in Markdown.
Using some plugins we cann add some new features to it.
As soon as a _new feature_ is added, it will be documented here.

## Colorboxes

We have the possibility to add boxes with a colored border and symbol to emphasize or highlight some important information.

> Example:
>
> ```markdown
> !!! info
>     This is an info box.
> ```

add a `""` after the keyword to **hide the title** of the box and only show the content.

There are the following options:

| Text                                  | Output                                                                       |                                                                         Output |                                 Text |
| :------------------------------------ | :--------------------------------------------------------------------------- | -----------------------------------------------------------------------------: | -----------------------------------: |
| `note`                                | ![note-block](../assets/images/avif/how-to-write-doc/note-block.png)         |     ![warning-block](../assets/images/avif/how-to-write-doc/warning-block.png) |             `warning` <br> `caution` |
| `summary` <br> `abstract` <br> `tldr` | ![summary-block](../assets/images/avif/how-to-write-doc/summary-block.png)   | ![attention-block](../assets/images/avif/how-to-write-doc/attention-block.png) |                          `attention` |
| `example` <br> `snippet`              | ![alt text](../assets/images/avif/how-to-write-doc/example-block.png)        |           ![fail-block](../assets/images/avif/how-to-write-doc/fail-block.png) | `fail` <br> `failure` <br> `missing` |
| `info`<br> `todo`                     | ![info-block](../assets/images/avif/how-to-write-doc/info-block.png)         |       ![danger-block](../assets/images/avif/how-to-write-doc/danger-block.png) |                             `danger` |
| `hint`<br> `tip`                      | ![hint-block](../assets/images/avif/how-to-write-doc/hint-block.png)         |         ![error-block](../assets/images/avif/how-to-write-doc/error-block.png) |                              `error` |
| `success` <br> `check` <br> `done`    | ![success-block](../assets/images/avif/how-to-write-doc/success-block.png)   |             ![bug-block](../assets/images/avif/how-to-write-doc/bug-block.png) |                                `bug` |
| `question` <br> `help` <br> `faq`     | ![question-block](../assets/images/avif/how-to-write-doc/question-block.png) |         ![quote-block](../assets/images/avif/how-to-write-doc/quote-block.png) |                  `quote` <br> `cite` |

