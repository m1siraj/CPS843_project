import cv2 
import numpy as np 
import datetime

import dash
from dash import Dash, dcc, html, Input, Output, State, callback, dash_table
from dash.dependencies import Input, Output
import dash_bootstrap_components as dbc

import base64
from io import BytesIO
from PIL import Image

# Sample Dash app with Bootstrap styles
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.CYBORG])
server = app.server

data = []
app.layout = html.Div(style={'alignItems': 'center', 'justifyContent': 'center', 'padding': '0 25%'}, children=[
    html.H5('Shape recognition and area calculations'),
    html.P('Select the type of image you are uploading:'),
    html.Div([
        dcc.Dropdown(['PNG', 'TIFF', 'JPEG', 'JPG'], 'PNG', id='demo-dropdown'),
        html.Div(id='dd-output-container')
    ]),
    html.Div([
    html.P('Upload the image in the dropdown below:'),
    dcc.Upload(
        id='upload-image',
        children=html.Div([
            'Drag and Drop or ',
            html.A('Select Files')
        ]),
        style={
            'width': '100%',
            'height': '60px',
            'lineHeight': '60px',
            'borderWidth': '1px',
            'borderStyle': 'dashed',
            'borderRadius': '5px',
            'textAlign': 'center',
            'margin': '10px'
        },
        # Allow multiple files to be uploaded
        multiple=False
    ),
    dbc.Button('Measure area', id='submit-val', className="me-2", n_clicks=0),
    dbc.Button('Detect shape', id='submit-val_1', className="me-2", n_clicks=0),
    html.Div([
        html.Div(id='output-image-upload', style={'display': 'flex', 'align': 'center', 'width': '800px', 'justify': 'center'}),
    ]),
    ]),
    html.Br(),
    html.Div([
        html.Div(id='prediction-output-header'),
        html.Div(id='prediction-output', style={'display': 'flex', 'align': 'center', 'width': '800px', 'justify': 'center'}),
        html.Br(),
        html.Div(id='prediction-output_1-header'),
        html.Div(id='prediction-output_1', style={'display': 'flex', 'align': 'center', 'width': '800px', 'justify': 'center'}),
    ])
])

def decode_image(contents):
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    return Image.open(BytesIO(decoded))

def cv2_to_dash_image(cv2_img):
    _, buffer = cv2.imencode('.png', cv2_img)
    encoded_img = base64.b64encode(buffer).decode('utf-8')
    return f'data:image/png;base64,{encoded_img}'

@app.callback(
    Output('output-image-upload', 'children'),
    [Input('upload-image', 'contents')],
    [State('upload-image', 'filename')]
)

def update_output(contents, filename):
    if contents is None:
        return html.Div()

    # Decode the uploaded image
    image = decode_image(contents)

    # Display the uploaded image in a smaller size
    image_style = {'width': '800px', 'padding': '2px', 'margin': '2px', 'border': '2px solid red', 'borderRadius': '15px'}
    output_image = html.Div([
        html.H5('Original image: '),
        html.Img(src=contents, style=image_style)])

    return output_image, filename

# Callback for dropdown
@callback(
    Output('dd-output-container', 'children'),
    Input('demo-dropdown', 'value')
)
def update_output(value):
    return f'You have selected {value}'

@app.callback(
     Output('prediction-output_1-header', 'children'),
     Output('prediction-output_1', 'children'),
    [Input('submit-val', 'n_clicks'),
     State('upload-image', 'contents'),
     State('upload-image', 'filename')
     ],
    prevent_initial_call=True
)
def update_output(n_clicks, contents, filename):
    if contents is None:
        return dbc.Alert("Please upload the image first", color="danger", dismissable=True),

    # Load the image 
    print(filename)
    img = cv2.imread(filename)
    print('something')
    # Convert the image to grayscale 
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 
    
    # Apply a threshold to the image to 
    # separate the objects from the background 
    ret, thresh = cv2.threshold( 
        gray, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU) 
    
    # Find the contours of the objects in the image 
    contours, hierarchy = cv2.findContours( 
        thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) 
    
    # Loop through the contours and calculate the area of each object 
    for cnt in contours: 
        area = cv2.contourArea(cnt) 
    
        # Draw a bounding box around each 
        # object and display the area on the image 
        x, y, w, h = cv2.boundingRect(cnt) 
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2) 
        cv2.putText(img, str(area), (x, y), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2) 
    
    #  Show the final image with the bounding boxes 
    # and areas of the objects overlaid on top 
    
    contours_image = cv2_to_dash_image(img)
    contours_image_style = {'width': '800px', 'padding': '2px', 'margin': '2px', 'border': '2px solid blue', 'borderRadius': '15px'}
    output_contours_image = html.Img(src=contours_image, style=contours_image_style)
    
    return html.Div(html.H4('Detected area Result: ')), output_contours_image


@app.callback(
    Output('prediction-output-header', 'children'),
     Output('prediction-output', 'children'),
    [Input('submit-val_1', 'n_clicks'),
     State('upload-image', 'contents'),
     State('upload-image', 'filename')
     ],
    prevent_initial_call=True
)
def update_output(n_clicks, contents, filename):
    if contents is None:
        return dbc.Alert("Please upload the image first", color="danger", dismissable=True),

    # reading image 
    img = cv2.imread(filename)
    
    # converting image into grayscale image 
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 
    
    # setting threshold of gray image 
    _, threshold = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY) 
    
    # using a findContours() function 
    contours, _ = cv2.findContours( 
        threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE) 
    
    i = 0
    
    # list for storing names of shapes 
    for contour in contours: 
    
        # here we are ignoring first counter because  
        # findcontour function detects whole image as shape 
        if i == 0: 
            i = 1
            continue
    
        # cv2.approxPloyDP() function to approximate the shape 
        approx = cv2.approxPolyDP( 
            contour, 0.01 * cv2.arcLength(contour, True), True) 
        
        # using drawContours() function 
        cv2.drawContours(img, [contour], 0, (0, 0, 255), 5) 
    
        # finding center point of shape 
        M = cv2.moments(contour) 
        if M['m00'] != 0.0: 
            x = int(M['m10']/M['m00']) 
            y = int(M['m01']/M['m00']) 
    
        # putting shape name at center of each shape 
        if len(approx) == 3: 
            cv2.putText(img, 'Triangle', (x, y), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2) 
    
        elif len(approx) == 4: 
            cv2.putText(img, 'Quadrilateral', (x, y), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2) 
    
        elif len(approx) == 5: 
            cv2.putText(img, 'Pentagon', (x, y), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2) 
    
        elif len(approx) == 6: 
            cv2.putText(img, 'Hexagon', (x, y), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2) 
    
        else: 
            cv2.putText(img, 'circle', (x, y), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2) 
  
    # Show the final image with the bounding boxes
    # and areas of the objects overlaid on top
    contours_image = cv2_to_dash_image(img)
    contours_image_style = {'width': '800px', 'border': '2px solid blue', 'borderRadius': '15px'}
    output_contours_image = html.Img(src=contours_image, style=contours_image_style)

    return html.Div(html.H4('Detected Shape Result: ')), output_contours_image


if __name__ == '__main__':
    app.run_server(debug=True)
