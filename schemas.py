INPUT_SCHEMA = {
    'prompt': {
        'type': str,
        'required': True,
    },

    'image':{
        'type': str,
        'required': True,
    },
    'aspect_ratio':{
        'type': str,
        'required': False,
        'default': 'match_input_image',
    },
    'seed': {
        'type': int,
        'required': False,
        'default': None,
    },
    'num_inference_steps':{
        'type': int,
        'required': False,
        'default': 28,
    },
    'guidance_scale':{
        'type': float,
        'required': False,
        'default': 2.5,
    }
}

