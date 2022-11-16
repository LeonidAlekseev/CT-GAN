import os
import re
import glob
import subprocess
from io import BytesIO
import zipfile
import uuid
import warnings
warnings.filterwarnings("ignore")

from flask import Flask, Blueprint, send_file
from flask_restx import Api, Resource, fields
from werkzeug.datastructures import FileStorage
from werkzeug.utils import secure_filename


BASE_DIR = os.path.dirname(os.path.realpath(__file__))
INTERPRETATOR_CMD = 'sudo /home/natitov/miniforge3/envs/ct_gan/bin/python'
INFERER_PATH = os.path.join(os.path.dirname(BASE_DIR), 'inferer.py')
TASK_PARAMS = {
    '3D Generation cancer': (
        '3d_ct_gan_cancer',
    ),
}


app = Flask(__name__)
app.secret_key = "20222022"
app.config['WEIGHTS_DIR'] = os.path.join(BASE_DIR, 'weights')
app.config['PREDICT_DIR'] = os.path.join(BASE_DIR, 'predict')
app.config['UPLOAD_DIR'] = os.path.join(BASE_DIR, 'upload')
app.config['ALLOWED_EXTENSIONS'] = ('.zip')
app.config['MAX_CONTENT_LENGTH'] = 1024**4

os.makedirs(app.config['WEIGHTS_DIR'], exist_ok=True)
os.makedirs(app.config['PREDICT_DIR'], exist_ok=True)
os.makedirs(app.config['UPLOAD_DIR'], exist_ok=True)

blueprint = Blueprint('api', __name__)
api = Api(blueprint, version='1.0', title='Prediction API',
    description='Predict generation for target')
app.register_blueprint(blueprint, url_prefix="/api")

upload_parser = api.parser()
upload_parser.add_argument('data', required=True, location='files',
    type=FileStorage)

predict_parser = api.parser()
predict_parser.add_argument('task', required=True, location='args',
    choices=list(TASK_PARAMS.keys()))
predict_parser.add_argument('data', required=True, location='args')
predict_parser.add_argument('locations', required=True, location='args')

export_parser = api.parser()
export_parser.add_argument('predict', required=True, location='args')


def check_extention(filename):
    is_supported = False
    for extention in app.config['ALLOWED_EXTENSIONS']:
        if filename.endswith(extention):
            is_supported = True
            break
    return is_supported


@api.route('/upload', doc={'description': 'Upload ZIP archive with DICOM'})
class Upload(Resource):
    @api.expect(upload_parser)
    @api.doc(params={
        'data': 'ZIP archive with DICOM',
    })
    @api.marshal_with(api.model('Upload', {
        'message': fields.String,
        'uuid': fields.String,
    }), code=201)
    def post(self):
        args = upload_parser.parse_args()
        data = args['data']
        if data.filename == '':
            return {
                'message' : 'No file selected for uploading',
            }, 400
        if not data or not check_extention(data.filename):
            return {
                'message' : f"Allowed file types are {app.config['ALLOWED_EXTENSIONS']}",
            }, 400
        data_uuid = str(uuid.uuid4())
        save_dir = os.path.join(app.config['UPLOAD_DIR'], data_uuid, 'dicom')
        os.makedirs(save_dir, exist_ok=True)
        data_object = data.stream._file  
        zip_object = zipfile.ZipFile(data_object)
        for el in zip_object.infolist():
            if re.match(r'*.dcm$', el.filename):
                zip_object.extract(el, path=save_dir)
        return {
            'message' : 'Successfully uploaded',
            'uuid': data_uuid,
        }, 201


@api.route('/predict', doc={'description': 'Load model and predict'})
class Predict(Resource):
    @api.expect(predict_parser)
    @api.doc(params={
        'locations': 'Locations coordinates (Example: x,y,z;x,y,z)',
        'data': 'Upload uuid',
        'task': 'Target task',
    })
    @api.marshal_with(api.model('Predict', {
        'message': fields.String,
        'uuid': fields.String,
    }), code=201)
    def post(self):
        args = predict_parser.parse_args()
        task = args['task']
        data = args['data']
        locations = args['locations']
        predict_uuid = str(uuid.uuid4())
        save_dir = os.path.join(app.config['PREDICT_DIR'], predict_uuid)
        os.makedirs(save_dir, exist_ok=True)
        try:
            weights_path = TASK_PARAMS[task]
            weights_path = os.path.join(app.config['WEIGHTS_DIR'], weights_path)
            data_path = os.path.join(app.config['UPLOAD_DIR'], data, 'dicom')
            result_path = save_dir
            visualization_path = os.path.join(result_path, 'visualization.png')
            subprocess.call(f'{INTERPRETATOR_CMD} {INFERER_PATH} -t "{task}" -w {weights_path} -d {data_path} -l {locations} -p {result_path} -v {visualization_path}', shell=True)
        except Exception as e:
            return {
                'message' : f"Inferer error {e}",
            }, 500
        return {
            'message' : 'Successfully predicted',
            'uuid': predict_uuid,
        }, 201


@api.route('/export', doc={'description': 'Export prediction'})
class Export(Resource):
    @api.expect(export_parser)
    @api.doc(params={
        'predict': 'Predict uuid',
    })
    def post(self):
        args = export_parser.parse_args()
        predict = args['predict']
        try:
            zip_file = BytesIO()
            predict_path = os.path.join(app.config['PREDICT_DIR'], predict)
            with zipfile.ZipFile(zip_file, 'w', zipfile.ZIP_DEFLATED) as zipf:
                for root, dirs, files in os.walk(predict_path):
                    for file in files:
                        source_name = os.path.join(root, file)
                        target_name = os.path.join(root.replace(BASE_DIR, ''), file)
                        zipf.write(source_name, target_name)
            zip_file.seek(0)
            return send_file(zip_file,
                attachment_filename=f"{predict}.zip",
                as_attachment=True)
        except Exception as e:
            return {
                'message' : f"Export error {e}",
            }, 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port='80', debug=True)