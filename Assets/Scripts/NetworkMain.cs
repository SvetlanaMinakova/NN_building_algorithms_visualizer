using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class NetworkMain : MonoBehaviour
{
    public List<LAYERTYPE> layer_types;
    public List<int> start_neurons_num;
    public List<int> adding_neurons_num;
    public List<float> dropout_perc;
    public List<float> neurons_adding_timewait;
    public List<List<int>> faded_neurons;

    public float start_Inc = 0.1f;
    public float max_Inc = 0.8f;
    public int Inc_step = 1;
    public float learning_rate = 0.01f;
    public MODE change_mode = MODE.NEURONSADDING;

    public GameObject MPL;
    public GameObject neuron_l0;
    public GameObject kernel_l0;
    public GameObject pooling_kernel_l0;
    public GameObject input_matrix;
    public GameObject input_kernel_l0;

    //inputs params
    /*int pixel_size = 10;
    int inp_w = 10;
    int inp_h = 10;*/
    //neurons adding params
    int m_delta_y = 20;
    int c_delta_y = 35;
    int c_delta_z = 35;
    public int stride = 3;
    //int c_delta_y = 20;
    int distance_between_layers = 200;
    private List<List<GameObject>> layers = new List<List<GameObject>>();
    //camera rotation
    private bool allow_scroll = false;
    public Transform target;
    public Vector3 offset;
    public float sensitivity = 3; // чувствительность мышки
    public float limit = 80; // ограничение вращения по Y
    public float zoom = 0.25f; // чувствительность при увеличении, колесиком мышки
    public float zoomMax = 10; // макс. увеличение
    public float zoomMin = 3; // мин. увеличение
    private float X, Y;
    //pause mode
    public bool paused;
    public bool paused_dropout;
    //input matrix fade pattern
    List<int> fade_pattern = new List<int>();
    bool input_faded = false;
    Random rand = new Random();

    // Use this for initialization
    void Start()
    {
        Init();
        //camera rotation
        limit = Mathf.Abs(limit);
        if (limit > 90) limit = 90;
        offset = new Vector3(offset.x, offset.y, -Mathf.Abs(zoomMax) / 2);
        transform.position = target.position + offset;
    }

    void Reset()
    {
        StopAllCoroutines();
        //network eraing
        GameObject layer;
        for (int l = layers.Count - 1; l >= 0; l--)
        {
            if (layers[l].Count > 0)
            {
                layer = layers[l][0].transform.parent.gameObject;
                Destroy(layer);
            }
        }
        layers.Clear();

        Init();

    }

    void Init()
    {
        //network pre-building
        if (change_mode == MODE.DEEPNETWORKS || change_mode== MODE.CONVOLUTIONALINCREMENT)
            StartCoroutine(AddLayersWithTimeOut());
        else
        {
            for (int i = 0; i < layer_types.Count; i++)
            {
                AddLayer(layer_types[i]);
                Addneurons(i, start_neurons_num[i] - 1);

                if (change_mode == MODE.NEURONSADDING && adding_neurons_num[i] > 0)
                    StartCoroutine(AddneuronsWithTimeout(i, neurons_adding_timewait[i], adding_neurons_num[i]));

                if (change_mode == MODE.DROPOUT && dropout_perc[i] > 0)
                    StartCoroutine(ImitateDropout(i, dropout_perc[i]));

            }
        }
        if (change_mode == MODE.GRADIENTVANISHING)
        {
            StartCoroutine(SendSignal(DIRECTION.FRONT, 0.5f));
        }

        if (change_mode == MODE.GRADIENTEXPLOSING)
        {
            StartCoroutine(SendSignal(DIRECTION.BACK, 1.5f, true));
        }
        if (change_mode == MODE.FRONTSENDING)
            StartCoroutine(SendSignal(DIRECTION.FRONT));
        if (change_mode == MODE.BACKSENDING)
            StartCoroutine(SendSignal(DIRECTION.BACK));
        //camera position??
        //   GetComponent<Camera>().transform.position = new Vector3((layer_types.Count - 1) * distance_between_layers/2, GetComponent<Camera>().transform.position.y, GetComponent<Camera>().transform.position.z);
    }

    // Update is called once per frame
    void Update()
    {
        if (allow_scroll)
        {
            if (Input.GetAxis("Mouse ScrollWheel") > 0) offset.z += zoom;
            else if (Input.GetAxis("Mouse ScrollWheel") < 0) offset.z -= zoom;
            offset.z = Mathf.Clamp(offset.z, -Mathf.Abs(zoomMax), -Mathf.Abs(zoomMin));

            X = transform.localEulerAngles.y + Input.GetAxis("Mouse X") * sensitivity;
            Y += Input.GetAxis("Mouse Y") * sensitivity;
            Y = Mathf.Clamp(Y, -limit, limit);
            transform.localEulerAngles = new Vector3(-Y, X, 0);
            transform.position = transform.localRotation * offset + target.position;
        }
        if (Input.GetKeyDown(KeyCode.Space))
        {
            if (!paused)
            {
                Time.timeScale = 0;
                paused = true;
            }
            else
            {
                Time.timeScale = 1;
                paused = false;
            }

        }

        if (Input.GetKeyDown(KeyCode.R))
        {
            Reset();
        }

        if (Input.GetKeyDown(KeyCode.A))
        {
            allow_scroll = !allow_scroll;
        }

        if (Input.GetKeyDown(KeyCode.D))
        {
          if (change_mode == MODE.CONVOLUTIONALINCREMENT)
                ImitateConvolutionalIncrement();

          else
            RemoveLinksOrNeurons();
        }


        if (Input.GetKeyDown(KeyCode.F))
        {
            if (!paused_dropout)
            {
                for (int i = 0; i < layer_types.Count; i++)
                {
                    if (layer_types[i] != LAYERTYPE.POOLING)
                    {
                        ApplyDropout(i, dropout_perc[i]);
                    }
                }
                paused_dropout = true;
            }

            else
            {

                for (int i = 0; i < layer_types.Count; i++)
                {
                    if (layer_types[i] != LAYERTYPE.POOLING)
                    {
                        ResetDropout(i);
                    }
                }
                paused_dropout = false;
            }
            
        }
        showonetoonepooling();

        if (!input_faded && layer_types[0]==LAYERTYPE.INPUT)
        {
            if(change_mode==MODE.CONVOLUTIONALINCREMENT)
            fade_input_connection();
            else 
                if(start_neurons_num.Count>0)
                {
                    showlocalvisionconvolutional();
                }
            input_faded = !input_faded;
        }
    }

    public void showlocalvisionconvolutional()
    {
        LineRenderer curlr;
        //int n_counter = 0;
        GameObject neuron;
        int inp_w_and_h = (int)Mathf.Sqrt(start_neurons_num[0]);
        int cells_num = inp_w_and_h / stride;
        int first_l_filters_num = start_neurons_num[1];
        int start_cell;
        int start_cell_row;
        int start_cell_col;
        List<int> cells_nums = new List<int>();
        fade_pattern.Clear();
        for (int i = 0; i < start_neurons_num[0]; i++)
        {
            //fill in possible visible cells
            start_cell_row = i / inp_w_and_h;
            start_cell_col = i - inp_w_and_h * start_cell_row + 1;
            if ((inp_w_and_h - start_cell_row + 1) - stride > 0 && (inp_w_and_h - start_cell_col + 1) - stride > 0)
                cells_nums.Add(i);
            //fill in fade pattern
            fade_pattern.Add(i);
        }

        int cell_id = 0;
        for (int c = 0; c < first_l_filters_num; c++)
        {
            cell_id = Random.Range(0, cells_nums.Count);
            start_cell = cells_nums[cell_id];

            for (int j = 0; j < stride; j++)
            {
                for (int i = 0; i < stride; i++)
                {
                    cell_id = start_cell + i + inp_w_and_h * j;
                    fade_pattern.Remove(cell_id);
                    cells_nums.Remove(cell_id);
                }

            }
        }
        //strict fade pattern
        /* int[] fade_pattern = { 2, 3, 4, 5, 6, 7, 12, 13, 14, 15, 16, 17, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 44, 45, 48, 49, 50, 51, 54, 55, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 82, 83, 84, 85, 86, 87, 92, 93, 94, 95, 96, 97 };*/
        foreach (var ind in fade_pattern)
        {
            neuron = layers[0][ind];
            if (neuron.transform.childCount > 0)
            {
                for (int j = 0; j < neuron.transform.childCount; j++)
                {
                    curlr = neuron.transform.GetChild(j).GetComponent<LineRenderer>();
                    if (curlr.startColor.a > 0.5)
                        StartCoroutine(Fade(curlr));
                }
            }
            // n_counter++;
        }

    }
    void ImitateConvolutionalIncrement()
    {
        int neurons_to_show_num = 0;
        faded_neurons = new List<List<int>>();
        for (int l = 0; l < layers.Count; l++)
        {
            faded_neurons.Add(new List<int>());
            if (layer_types[l] == LAYERTYPE.CONVOLUTIONAL || layer_types[l] == LAYERTYPE.POOLING)
            {
                neurons_to_show_num = (int)((float)start_neurons_num[l] * start_Inc);
                //min 1 neuron to show
                if (neurons_to_show_num == 0)
                    neurons_to_show_num++;
                //show input connection with visible neurons
                for (int i = 0; i < neurons_to_show_num; i++)
                    show_input_pattern_convolutional(i);
                LineRenderer curlr;
                for (int i = start_neurons_num[l] - 1; i >= neurons_to_show_num; i--)
                {
                    Fade(layers[l][i]);
                    faded_neurons[l].Add(i);
                    Debug.Log("neuron " + i + " faded");

                    foreach (var neuron in layers[l-1])
                    {
                        curlr = neuron.transform.GetChild(i).GetComponent<LineRenderer>();
                        StartCoroutine(Fade(curlr));
                    }

                    if (l != layer_types.Count - 1)
                    {
                        if (layer_types[l + 1] == LAYERTYPE.MPL || layer_types[l + 1] == LAYERTYPE.CONVOLUTIONAL)
                        {
                            for (int j = 0; j < start_neurons_num[l + 1]; j++)
                            {
                                curlr = layers[l][i].transform.GetChild(j).GetComponent<LineRenderer>();
                                StartCoroutine(Fade(curlr));
                            }
                            StartCoroutine(ImitateNeuronsIncrement(l));
                        }
                    }
                }
            }
        }
    }


    void LateUpdate()
    {
        RenewLinks(layers);
    }


    public IEnumerator ChangeColor(GameObject neuron)
    {
        if (neuron)
        {
            color_prior cp = new color_prior(Color.gray, 0, true, 1);
            while (cp.continue_proc)
            {
                yield return new WaitForSeconds(0.1f);
                switch (change_mode)
                {
                    case (MODE.USUAL):
                        cp = ChangeColorUsual(neuron, cp);
                        break;

                    case (MODE.NEURONSADDING):
                        cp = ChangeColorUsual(neuron, cp);
                        break;

                    case (MODE.NEURONSREMOVING):
                        cp = ChangeColorNeuronsRemoving(neuron, cp);
                        break;

                    case (MODE.DROPOUT):
                       // cp = ChangeColorDropout(neuron, cp);
                        break;

                    case (MODE.WEIGHTDECAY):
                        cp = ChangeColorWeightDecay(neuron, cp);
                        break;

                    default:
                        break;

                }
            }
        }
    }


    private IEnumerator SendSignal(DIRECTION direction, float k = 0, bool is_gradient_exploding = false)
    {
        int layer_id = 0;
        LineRenderer curlr;
        //bool send_front = true; //for front_and_back sending
        //clear lines' color
        while (true)
        {
            foreach (List<GameObject> layer in layers)
            {
                foreach (GameObject neuron in layer)
                {
                    for (int j = 0; j < neuron.transform.childCount; j++)
                    {
                        curlr = neuron.transform.GetChild(j).GetComponent<LineRenderer>();
                        curlr.startColor = Color.black;
                        curlr.endColor = Color.black;
                    }
                }
            }

            //show sending signal
            foreach (List<GameObject> layer in layers)
            {
                switch (direction)
                {
                    case (DIRECTION.FRONT):
                        yield return new WaitForSeconds(0.2f * layer_id);
                        foreach (GameObject neuron in layers[layer_id])
                            StartCoroutine(SendNeuronSignalFront(neuron, layer_id, k));
                        break;

                    case (DIRECTION.BACK):
                        yield return new WaitForSeconds(0.1f * (layers.Count - layer_id - 1));
                        foreach (GameObject neuron in layers[layers.Count - layer_id - 1])
                            StartCoroutine(SendNeuronSignalBack(neuron, layers.Count - layer_id - 1,k, is_gradient_exploding));
                        break;

                    default:
                        break;
                }
                layer_id++;
            }
            layer_id = 0;
        }
       
    }


    private IEnumerator SendNeuronSignalFront(GameObject neuron, int layer_id, float vanishing_k = 0)
    {
        //cp.color start_color, end_color=1-startcolor, delta=0.1f
        LineRenderer curlr;
        Color c2 = Color.black;
        color_prior cp = new color_prior(Color.black, 0, true, 1);
        float delta_g = 0.1f;
        float brightness_perc = Mathf.Pow((1 - vanishing_k), layer_id);
        float max_brightness = brightness_perc;

       while (cp.continue_proc)
        {
            yield return new WaitForSeconds(0.1f);

            for (int j = 0; j < neuron.transform.childCount; j++)
            {
                curlr = neuron.transform.GetChild(j).GetComponent<LineRenderer>();

                switch (cp.prior)
                {
                    case (0):
                        if (cp.color.g < max_brightness)
                        {
                            cp.color.g += delta_g * max_brightness;
                        }
                        else
                        {
                            cp.prior = 1;
                        }
                        break;

                    case (1):

                        if (cp.color.g > 0)
                        {
                            cp.color.g -= delta_g * max_brightness;
                            c2.g += delta_g * max_brightness;
                        }
                        else
                        {
                            cp.color = Color.black;
                            c2 = Color.black;
                            cp.continue_proc = false;

                        }
                        break;
                }


                curlr.startColor = cp.color;
                curlr.endColor = c2;
            }
        }
    }

    private IEnumerator SendNeuronSignalBack(GameObject neuron, int layer_id,float explosing_k, bool is_gradient_exploding = false)
    {
        //cp.color start_color, end_color=1-startcolor, delta=0.1f
        LineRenderer curlr;
        Color c2 = Color.black;
        color_prior cp = new color_prior(Color.black, 0, true, 1);
        float delta = 0.1f;
        float max_brightness = 1;
        explosing_k = Mathf.Pow(explosing_k, layer_id);

        while (cp.continue_proc)
        {
            yield return new WaitForSeconds(0.1f);

            for (int j = 0; j < neuron.transform.childCount; j++)
            {
                curlr = neuron.transform.GetChild(j).GetComponent<LineRenderer>();

                switch (cp.prior)
                {
                    case (0):
                        if (cp.color.r < max_brightness)
                        {
                            cp.color.r += delta;
                        }
                        else
                        {
                            cp.prior = 1;
                        }
                        break;

                    case (1):

                        if (cp.color.r > 0)
                        {
                            cp.color.r -= delta*explosing_k;
                            c2.r += delta*explosing_k;
                        }
                        else
                        {
                            if (!is_gradient_exploding)
                            {
                                curlr.startColor = Color.black;
                                curlr.endColor = Color.black;
                                cp.color = Color.black;
                            }
                            cp.continue_proc = false;
                        }
                     
                        break;
                }

                curlr.endColor = cp.color;
                curlr.startColor = c2;
            }
        }
    }

    private color_prior ChangeColorUsual(GameObject neuron, color_prior cp)
    {
        LineRenderer curlr;
        cp = GetNewColor(cp);
        //change neuron color the same way as links??
        //neuron.GetComponent<Renderer>().material.color = cp.color;
        for (int j = 0; j < neuron.transform.childCount; j++)
        {
            curlr = neuron.transform.GetChild(j).GetComponent<LineRenderer>();
            curlr.startColor = cp.color;
            curlr.endColor = cp.color;
        }
        return cp;
    }

    private color_prior ChangeColorDropout(GameObject neuron, color_prior cp)
    {
        LineRenderer curlr;
        float no_dropouted_links_perc = 0;
        Vector3 mid_color = new Vector3(0, 0, 0);
        for (int j = 0; j < neuron.transform.childCount; j++)
        {
            curlr = neuron.transform.GetChild(j).GetComponent<LineRenderer>();
            if (curlr.startColor.a > 0.5)
            {
                no_dropouted_links_perc++;
                cp.color = curlr.startColor;
                cp = GetNewColor(cp);
                curlr.startColor = cp.color;
                curlr.endColor = cp.color;
            }
        }
        //neuron's color change depends on it's dropouted links num
        no_dropouted_links_perc /= neuron.transform.childCount;
        //cp.color = curlr.startColor;
        cp.rate = no_dropouted_links_perc;
        cp = GetNewColor(cp);
       // neuron.GetComponent<Renderer>().material.color = cp.color;
        return cp;
    }

    private color_prior ChangeColorWeightDecay(GameObject neuron, color_prior cp)
    {
        LineRenderer curlr;
        Color col;
        for (int j = 0; j < neuron.transform.childCount; j++)
        {
            curlr = neuron.transform.GetChild(j).GetComponent<LineRenderer>();
            if (Random.Range(1, 100) % 3 == 0)
                col = Color.red;
            else
                col = Color.black;

            curlr.startColor = col;
            curlr.endColor = col;
        }
        cp.continue_proc = false;
        return cp;
    }


    private color_prior ChangeColorNeuronsRemoving(GameObject neuron, color_prior cp)
    {
        LineRenderer curlr;
        Color col;
        if (neuron.transform.childCount > 0)
        {
            if (Random.Range(1, 100) % 3 == 0)
            {
                col = Color.red;
                neuron.GetComponent<Renderer>().material.color = col;
            }
            else
                col = Color.black;
            for (int j = 0; j < neuron.transform.childCount; j++)
            {
                curlr = neuron.transform.GetChild(j).GetComponent<LineRenderer>();
                curlr.startColor = col;
                curlr.endColor = col;
            }
        }
        cp.continue_proc = false;
        return cp;
    }

    public color_prior GetNewColor(color_prior cp)
    {
        switch (cp.prior)
        {
            case (0):
                if (cp.color.r > learning_rate * cp.rate)
                {
                    cp.color.r -= learning_rate * cp.rate;
                    cp.color.b -= learning_rate * cp.rate;
                }
                else
                    cp.prior = 1;
                return cp;

            case (1):
                if (cp.color.r < (1 - learning_rate * cp.rate))
                {
                    cp.color.r += learning_rate * cp.rate;
                }
                else
                    cp.prior = 2;
                return cp;

            case (2):
                if (cp.color.g > learning_rate * cp.rate)
                {
                    cp.color.g -= learning_rate * cp.rate;
                    cp.color.b -= learning_rate * cp.rate;
                }
                else
                    cp.prior = 3;
                return cp;

            default:
                cp.continue_proc = false;
                return cp;
        }

    }

    public void RemoveLinksOrNeurons()
    {
        if (change_mode == MODE.WEIGHTDECAY)
            RemoveLinksWeightDecay();
        if (change_mode == MODE.NEURONSREMOVING)
            RemoveNeuronsWeightDecay();
    }

    public void RemoveLinksWeightDecay()
    {
        LineRenderer curlr;
        int l_counter = 0;
        int n_counter = 0;
        foreach (List<GameObject> layer in layers)
        {
            //redirect child's links
            foreach (GameObject neuron in layer)
            {
                if (neuron.transform.childCount > 0)
                {
                    for (int j = 0; j < neuron.transform.childCount; j++)
                    {
                        curlr = neuron.transform.GetChild(j).GetComponent<LineRenderer>();
                        if (curlr.startColor.r == 1)
                            StartCoroutine(Fade(curlr));
                    }

                }
                n_counter++;
            }
            l_counter++;
        }

    }

    public void RemoveNeuronsWeightDecay()
    {
        List<int> neurons_to_delete = new List<int>();
        for (int j = 0; j < layers.Count - 1; j++)
        {
            for (int i = 0; i < layers[j].Count; i++)
            {
                if (layers[j][i].GetComponent<Renderer>().material.color == Color.red)
                {
                    neurons_to_delete.Add(i);
                }
            }
            StartCoroutine(DeleteneuronsWithTimeout(j, 0.01f, neurons_to_delete));
            neurons_to_delete.Clear();
        }
    }


    public void showonetoonepooling()
    {
        LineRenderer curlr;
        int l_counter = 0;
        int n_counter = 0;
        GameObject neuron;
        for (int i=0;i<layer_types.Count;i++)
        {
            if (layer_types[i] == LAYERTYPE.POOLING && i > 0 && (layer_types[i - 1] == LAYERTYPE.CONVOLUTIONAL||layer_types[i + 1] == LAYERTYPE.CONVOLUTIONAL))
                for (int n = 0; n < layers[i - 1].Count;n++ )
                {
                    if(layer_types[i - 1] == LAYERTYPE.CONVOLUTIONAL)
                    neuron = layers[i - 1][n];
                    else
                        neuron = layers[i][n];
                    if (neuron.transform.childCount > 0)
                    {
                        for (int j = 0; j < neuron.transform.childCount; j++)
                        {
                            curlr = neuron.transform.GetChild(j).GetComponent<LineRenderer>();
                            if (j!=n && curlr.startColor.a > 0.5)
                                StartCoroutine(Fade(curlr));
                        }

                    }
                    n_counter++;
                }
            l_counter++;
        }

    }

    public void fade_input_connection()
    {
        fade_pattern = new List<int>();
        LineRenderer curlr;
        int first_l_filters_num = start_neurons_num[1];
        int n_counter = 0;
        foreach (var neuron in layers[0])
        {
            if (neuron.transform.childCount > 0)
            {
                for (int j = 0; j < neuron.transform.childCount; j++)
                {
                    curlr = neuron.transform.GetChild(j).GetComponent<LineRenderer>();
                    if (curlr.startColor.a > 0.5)
                        StartCoroutine(Fade(curlr));
                }
            }
            fade_pattern.Add(n_counter);
            n_counter++;
        }

        Debug.Log("fade pattern count: " + fade_pattern.Count);
    }

   public void show_input_pattern_convolutional(int conv_filter_id)
   {
       LineRenderer curlr;
       int n_counter = 0;
       GameObject neuron;
       int inp_w_and_h = (int)Mathf.Sqrt(start_neurons_num[0]);
       int cells_num = inp_w_and_h / stride;

       int first_l_filters_num = start_neurons_num[1];
       int start_cell;
       int start_cell_row;
       int start_cell_col;
       List<int> cells_nums = new List<int>();
       List<int> visible_pattern = new List<int>();
       int cur_id;

       for (int i = 0; i < fade_pattern.Count; i++)
       {
           cur_id = fade_pattern[i];
           //fill in possible visible cells
           start_cell_row = cur_id / inp_w_and_h;
           start_cell_col = cur_id - inp_w_and_h * start_cell_row + 1;
           if ((inp_w_and_h - start_cell_row + 1) - stride > 0 && (inp_w_and_h - start_cell_col + 1) - stride > 0)
               cells_nums.Add(cur_id);
       }

       int cell_id = 0;

           cell_id = Random.Range(0, cells_nums.Count);
           start_cell = cells_nums[cell_id];

           for (int j = 0; j < stride; j++)
           {
               for (int i = 0; i < stride; i++)
               {
                   cell_id = start_cell + i + inp_w_and_h * j;
                   fade_pattern.Remove(cell_id);
                   visible_pattern.Add(cell_id);
               }

           }
       
       foreach (var ind in visible_pattern)
       {
           neuron = layers[0][ind];
           if (neuron.transform.childCount > 0)
           {
               curlr = neuron.transform.GetChild(conv_filter_id).GetComponent<LineRenderer>();
               if (curlr.startColor.a < 0.5)
               StartCoroutine(Appear(curlr));
           }
           n_counter++;
       }

   }

   public void RedrawLinks(List<List<GameObject>> container)
    {
        LineRenderer curlr;
        int l_counter = 0;
        int n_counter = 0;
        foreach (List<GameObject> layer in container)
        {
            //redirect child's links
            foreach (GameObject neuron in layer)
            {
                if (neuron.transform.childCount > 0)
                {
                    for (int j = 0; j < neuron.transform.childCount; j++)
                    {
                        curlr = neuron.transform.GetChild(j).GetComponent<LineRenderer>();
                        if (curlr.startColor.a < 0.5)
                            StartCoroutine(Appear(curlr));
                        if (curlr.startColor.a > 0.5)
                            StartCoroutine(Fade(curlr));
                    }

                }
                n_counter++;
            }
            l_counter++;
        }

    }

    public void RenewLinks(List<List<GameObject>> container)
    {
        LineRenderer curlr;
        int l_counter = 0;
        List<GameObject> layer;
        GameObject neuron;
        for (int l = 0; l < layers.Count; l++)
        {
            layer = container[l];
            //redirect child's links
            for (int n = 0; n < layer.Count; n++)
            {
                neuron = layer[n];
                if (neuron.transform.childCount > 0)
                {
                    for (int j = 0; j < neuron.transform.childCount; j++)
                    {
                        curlr = neuron.transform.GetChild(j).GetComponent<LineRenderer>();
                        try
                        {
                            curlr.SetPosition(0, neuron.transform.position);
                            curlr.SetPosition(1, container[l_counter + 1][j].transform.position);
                        }
                        catch (System.Exception)
                        {
                            Destroy(curlr.gameObject); 
                        }
                    }
                }
            }
            l_counter++;
        }
    }


    private IEnumerator AddneuronsWithTimeout(int layer_id, float timewait, int max_adding_neurons_number)
    {
        string n_type_name = "n";
        float delta_y = m_delta_y;
        List<GameObject> container = layers[layer_id];
        GameObject parent_neuron = container[0];
        LineRenderer curlr;
        int id;
        for (int i = 0; i < max_adding_neurons_number; i++)
        {
            yield return new WaitForSeconds(timewait);
            id = container.Count;
            var new_neuron = Instantiate(parent_neuron);
            new_neuron.name = n_type_name + layer_id + (id + 1).ToString();
            new_neuron.transform.parent = parent_neuron.transform.parent.transform;

            new_neuron.transform.position = GetNeuronPosition(layer_id, id);
            //redirect child's links
            if (new_neuron.transform.childCount > 0)
            {
                for (int j = 0; j < new_neuron.transform.childCount; j++)
                {
                    curlr = new_neuron.transform.GetChild(j).GetComponent<LineRenderer>();
                    curlr.SetPosition(0, new_neuron.transform.position);
                }
            }
            //link with neurons of prev layer
            if (layer_id > 0)
            {
                foreach (var n in layers[layer_id - 1])
                {
                    Addlink(n, new_neuron);
                }
            }

            new_neuron.GetComponent<Renderer>().material.color = GetNeuronColor(layer_id);
            StartCoroutine(ChangeColor(new_neuron));
            container.Add(new_neuron);
        }
    }


    private IEnumerator DeleteneuronsWithTimeout(int layer_id, float timewait, List<int> neuron_ids = null)
    {
        if (neuron_ids == null)
        {
            neuron_ids = new List<int>();
            for (int i = 0; i < neuron_ids.Count; i++)
                neuron_ids.Add(i);
        }
        int neurons_removed = 0;
        for (int i = 0; i < neuron_ids.Count; i++)
        {
            List<GameObject> container = layers[layer_id];
            GameObject neuron = container[neuron_ids[i] - neurons_removed];
            yield return new WaitForSeconds(timewait);
            container.Remove(neuron);
            Destroy(neuron);
            neurons_removed++;
        }

    }

    private void Deleteneurons(int layer_id, List<int> neuron_ids = null)
    {
        if (neuron_ids == null)
        {
            neuron_ids = new List<int>();
            for (int i = 0; i < neuron_ids.Count; i++)
                neuron_ids.Add(i);

        }
        int neurons_removed = 0;
        for (int i = 0; i < neuron_ids.Count; i++)
        {
            List<GameObject> container = layers[layer_id];
            GameObject neuron = container[neuron_ids[i] - neurons_removed];
            container.Remove(neuron);
            Destroy(neuron);
            neurons_removed++;
        }

    }



    void Addneurons(int layer_id, int max_adding_neurons_number)
    {
        string n_type_name = "n";
        float delta_y = m_delta_y;

        List<GameObject> container = layers[layer_id];
        GameObject parent_neuron = container[0];
        LineRenderer curlr;
        int id;
        for (int i = 0; i < max_adding_neurons_number; i++)
        {
            id = container.Count;
            var new_neuron = Instantiate(parent_neuron);
            new_neuron.name = n_type_name + layer_id + (id + 1).ToString();
            new_neuron.transform.parent = parent_neuron.transform.parent.transform;

            new_neuron.transform.position = GetNeuronPosition(layer_id, id);
            //redirect child's links
            if (new_neuron.transform.childCount > 0)
            {
                for (int j = 0; j < new_neuron.transform.childCount; j++)
                {
                    curlr = new_neuron.transform.GetChild(j).GetComponent<LineRenderer>();
                    curlr.SetPosition(0, new_neuron.transform.position);
                }
            }
            //link with neurons of prev layer
            if (layer_id > 0)
            {

                foreach (var n in layers[layer_id - 1])
                {
                    Addlink(n, new_neuron);
                }
            }

            new_neuron.GetComponent<Renderer>().material.color = GetNeuronColor(layer_id);
            StartCoroutine(ChangeColor(new_neuron));
            if (layer_id > 0)
            {
                if (layer_types[layer_id] == LAYERTYPE.POOLING && layer_types[layer_id - 1] == LAYERTYPE.CONVOLUTIONAL)
                    StopCoroutine(ChangeColor(new_neuron));
            }

            container.Add(new_neuron);
        }
    }


    private Vector3 GetNeuronPosition(int layer_id,int n_id)
    {
        GameObject parent_neuron = layers[layer_id][0];
        float new_n_x = parent_neuron.transform.position.x;
        float new_n_y = parent_neuron.transform.position.y;
        float new_n_z = parent_neuron.transform.position.z;

        if(layer_types[layer_id]==LAYERTYPE.INPUT)
        {
            int row_len = (int)Mathf.Sqrt(start_neurons_num[0]);
            Debug.Log("row len " + row_len);
            int row = n_id / row_len;
            int num_in_row = n_id % row_len;
            int half_len = row_len / 2;

            if (n_id < half_len)
                new_n_y += row * c_delta_y;
            else
                new_n_y -= row * c_delta_y;

            if (num_in_row < half_len)
                new_n_z = num_in_row * c_delta_z;
            else
                new_n_z = num_in_row * c_delta_z;

            new_n_y += c_delta_y * half_len;
            new_n_z -= c_delta_z * half_len;

            if (n_id == start_neurons_num[0] - 1)
            {
                Vector3 n00_pos = layers[0][0].transform.position;
                n00_pos.y += c_delta_y * half_len;
                n00_pos.z -= c_delta_z * half_len;
                layers[0][0].transform.position=n00_pos;
            }

            return new Vector3(new_n_x, new_n_y, new_n_z);
        }


        //growing symmetrical
        if (n_id % 2 == 0)
            new_n_y = n_id * m_delta_y;
        else
            new_n_y = -n_id * m_delta_y;

        return new Vector3(new_n_x, new_n_y, new_n_z);
    }

    private Color GetNeuronColor(int layer_id)
    {
        if (layer_id == 0)
            return Color.yellow;

        //orange color
        if (layer_id == layer_types.Count - 1)
            return new Color(1,0.5f, 0);

        if (layer_types[layer_id] == LAYERTYPE.CONVOLUTIONAL)
            return Color.magenta;

        if (layer_types[layer_id] == LAYERTYPE.POOLING)
            return Color.cyan;

        return Color.green;
    }



    private void Addlink(GameObject n1, GameObject n2)
    {
        GameObject newlink = new GameObject();
        newlink.name = "l_" + n1.name + "_" + n2.name;
        newlink.transform.parent = n1.transform;
        newlink.AddComponent<LineRenderer>();
        LineRenderer line = newlink.GetComponent<LineRenderer>();
        line.SetPosition(0, n1.transform.position);
        line.SetPosition(1, n2.transform.position);
        line.material = new Material(Shader.Find("Particles/Alpha Blended"));
        Color c = Color.black;
        c.a = 0;
        line.startColor = c;
        line.endColor = c;
        StartCoroutine(Appear(line));
    }

    private IEnumerator Appear(LineRenderer line)
    {
        for (float f = 0; f <= 1f; f += 0.1f)
        {
            yield return new WaitForSeconds(0.1f);
            if (line)
            {
                Color c = line.startColor;
                c.a = f;
                line.startColor = c;
                line.endColor = c;
            }
        }
    }

    private void Activate(int neuron_id, int layer_id)
    {
        //show neuron to bhe active
        Appear(neuron_id, layer_id);
        LAYERTYPE l_type=layer_types[layer_id];
        LineRenderer curlr;
        //show input links to be active
        if(layer_id<layer_types.Count-1)
        {
            LAYERTYPE next_l_type = layer_types[layer_id+1];
            if (l_type==LAYERTYPE.CONVOLUTIONAL && next_l_type==LAYERTYPE.POOLING)
            {
                curlr = layers[layer_id][neuron_id].transform.GetChild(neuron_id).GetComponent<LineRenderer>();
                    StartCoroutine(Appear(curlr));
            }
        }
        if (layer_id == 1 && layer_types[0] == LAYERTYPE.INPUT)
        {
            show_input_pattern_convolutional(neuron_id);
        }
    }

    private void Appear(int neuron_id,int layer_id)
    { GameObject neuron = layers[layer_id][neuron_id];
        if (neuron)
        { 
            Color c = GetNeuronColor(layer_id);
            neuron.GetComponent<Renderer>().material.color = c;
        }
        faded_neurons[layer_id].Remove(neuron_id);
    }

    private IEnumerator Fade(LineRenderer line)
    {
        for (float f = 1f; f >= 0; f -= 0.1f)
        {
          
            if (line)
            {
                Color c = line.startColor;
                c.a = f;
                line.startColor = c;
                line.endColor = c;
            }
            yield return new WaitForSeconds(0.1f);
        }
    }

    private void Fade(GameObject neuron)
    {
            if (neuron)
            {
                Color c = Color.gray;
                neuron.GetComponent<Renderer>().material.color = c;
            }
    }


    private IEnumerator ImitateDropout(int layer_id, float dropout_percentage)
    {
        for (int i = 0; i < 100; i++)
        {
            Random.InitState((int)System.DateTime.Now.Ticks);
            yield return new WaitForSeconds(0.01f * layers[0].Count);
            if (i % 2 == 0)
                ApplyDropout(layer_id, dropout_percentage);
            else
                ResetDropout(layer_id);
        }
    }

    private void ApplyDropout(int layer_id, float dropout_percentage)
    {
        List<List<int>> dropouted_lines = new List<List<int>>();
        float dropout_lines_num;
        LineRenderer curlr;
        int n_counter = 0;
        foreach (var neuron in layers[layer_id])
        {
            dropouted_lines.Add(new List<int>());
            dropout_lines_num = (float)neuron.transform.childCount * dropout_percentage;
            for (int i = 0; i < dropout_lines_num; i++)
            {
                dropouted_lines[n_counter].Add(Random.Range(0, neuron.transform.childCount));
            }
            for (int i = 0; i < dropout_lines_num; i++)
            {
                curlr = neuron.transform.GetChild(dropouted_lines[n_counter][i]).GetComponent<LineRenderer>();
                StartCoroutine(Fade(curlr));
            }
            n_counter++;
        }
    }

    private void ResetDropout(int layer_id)
    {
        LineRenderer curlr;
        //redirect child's links
        foreach (GameObject neuron in layers[layer_id])
        {
            if (neuron.transform.childCount > 0)
            {
                for (int j = 0; j < neuron.transform.childCount; j++)
                {
                    curlr = neuron.transform.GetChild(j).GetComponent<LineRenderer>();
                    if (curlr.startColor.a < 0.5)
                        StartCoroutine(Appear(curlr));
                }
            }
        }
    }

    private IEnumerator ImitateNeuronsIncrement(int layer_id)
    {
        faded_neurons[layer_id].Sort();
        faded_neurons[layer_id - 1].Sort();
        for (int i = 0; i < faded_neurons[layer_id].Count; i++)
        {
            yield return StartCoroutine(ImitateLinesIncrement(layer_id));
            if (faded_neurons[layer_id].Count > 0)
            {
                //Activate neurons of prev layer
                Activate(faded_neurons[layer_id][0],layer_id - 1);
                //start Increment
                Appear(faded_neurons[layer_id][0], layer_id);
            }
        }
    }
    private IEnumerator ImitateLinesIncrement(int layer_id)
    {
        float dropout_percentage = (1-start_Inc);
        bool lines_not_dropouted = true;
        bool lines_not_showed_again = true;
       
        while(lines_not_dropouted || lines_not_showed_again)
        {
            Random.InitState((int)System.DateTime.Now.Ticks);
            yield return new WaitForSeconds(0.4f);
            if (lines_not_dropouted)
            {
                ApplyDropoutIncremental(layer_id, dropout_percentage);
                lines_not_dropouted = false;
            }
            else
               lines_not_showed_again = ResetDropoutIncremental(layer_id);
        }
    }


    private void ApplyDropoutIncremental(int layer_id, float dropout_percentage)
    {
        List<List<int>> dropouted_lines = new List<List<int>>();
        float dropout_lines_num;
        LineRenderer curlr;
        int n_counter = 0;
        GameObject neuron;
        int showed_neurons_num = start_neurons_num[layer_id] - faded_neurons[layer_id].Count;
        for (int sn = 0; sn < showed_neurons_num; sn++)
        {
            neuron = layers[layer_id][sn];
            dropouted_lines.Add(new List<int>());
            dropout_lines_num = (float)neuron.transform.childCount * dropout_percentage;
            for (int i = 0; i < dropout_lines_num; i++)
            {
                dropouted_lines[n_counter].Add(Random.Range(0, neuron.transform.childCount));
            }
            for (int i = 0; i < dropout_lines_num; i++)
            {
                curlr = neuron.transform.GetChild(dropouted_lines[n_counter][i]).GetComponent<LineRenderer>();
                if(curlr.startColor.a>0.5)
                    StartCoroutine(Fade(curlr));
            }
            n_counter++;
        }
    }

    private bool ResetDropoutIncremental(int layer_id)
    {
        LineRenderer curlr;
        //redirect child's links
        GameObject neuron;
        
        int new_showed_lines = 0;
        int all_showed_lines=0;
        int showed_neurons_num = start_neurons_num[layer_id]-faded_neurons[layer_id].Count;
        int all_lines = showed_neurons_num * (start_neurons_num[layer_id - 1] - faded_neurons[layer_id - 1].Count);
        float showed_lines_perc = 0;

        for(int i=0; i<showed_neurons_num;i++)
        {
            neuron = layers[layer_id][i];
            if (neuron.transform.childCount > 0)
            {
                for (int j = 0; j < neuron.transform.childCount; j++)
                {
                    if (!faded_neurons[layer_id+1].Contains(j))
                    {
                        curlr = neuron.transform.GetChild(j).GetComponent<LineRenderer>();
                        if (curlr.startColor.a < 0.5)
                        {
                            StartCoroutine(Appear(curlr));
                            new_showed_lines++;
                            if (new_showed_lines == Inc_step)
                                return true;
                        }
                        else
                        {
                           // all_showed_lines++;
                           // if (((float)all_showed_lines / (float)all_lines) >= max_Inc)
                             //   return false;
                        }
                    }
                }
            }
        }
        return false;
    }

    public IEnumerator AddLayersWithTimeOut()
    {
        for (int i = 0; i < layer_types.Count; i++)
        {
            AddLayer(layer_types[i]);
            Addneurons(i, start_neurons_num[i] - 1);
            yield return new WaitForSeconds(0.5f);
        }
       // StartCoroutine(SendSignal(DIRECTION.FRONT));
    }



    void AddLayer(LAYERTYPE layer_type)
    {
        int layer_id = layers.Count;
        GameObject new_l = new GameObject();
        new_l.name = "Layer" + layer_id;
        //neurons of layer container
        layers.Add(new List<GameObject>());
        new_l.transform.parent = MPL.transform;
        new_l.transform.position = new Vector3(layer_id * distance_between_layers,
            new_l.transform.position.y, new_l.transform.position.z);
        //add first neuron to layer
        GameObject new_neuron = InstantiateNeuron(layer_type);
        new_neuron.name = "n" + layer_id + "0";
        new_neuron.transform.parent = new_l.transform;
        new_neuron.transform.position = new Vector3(new_l.transform.position.x, new_l.transform.position.y, new_l.transform.position.z);
        new_neuron.GetComponent<Renderer>().material.color = GetNeuronColor(layer_id);

        layers[layer_id].Add(new_neuron);

            StartCoroutine(ChangeColor(new_neuron));



        if (layer_id > 0)
        {
            if (layer_types[layer_id] == LAYERTYPE.POOLING && layer_types[layer_id - 1] == LAYERTYPE.CONVOLUTIONAL)
                StopCoroutine(ChangeColor(new_neuron));

            foreach (var n in layers[layer_id - 1])
            {
                Addlink(n, new_neuron);
            }
        }

    }

    private GameObject InstantiateNeuron(LAYERTYPE l_type)
    {
        switch (l_type)
        {
            case (LAYERTYPE.MPL):
                return Instantiate(neuron_l0);

            case (LAYERTYPE.CONVOLUTIONAL):
                return Instantiate(kernel_l0);

            case (LAYERTYPE.POOLING):
                return Instantiate(pooling_kernel_l0);

            case(LAYERTYPE.INPUT):
                return Instantiate(input_kernel_l0);
            default:
                return Instantiate(neuron_l0);
        }
    }

    public enum LAYERTYPE
    {
        MPL,
        CONVOLUTIONAL,
        POOLING,
        INPUT,
        DECONVOLUTIONAL
    }

    private enum LINKTYPE
    {
        FULLYCONNECTED,
        ONETOONE
    }

    public enum MODE
    {
        USUAL,
        NONE,
        WEIGHTDECAY,
        DROPOUT,
        NEURONSADDING,
        NEURONSREMOVING,
        GRADIENTVANISHING,
        GRADIENTEXPLOSING,
        FRONTSENDING,
        BACKSENDING,
        DEEPNETWORKS,
        CONVOLUTIONALINCREMENT
    }

    public enum DIRECTION
    {
        FRONT,
        BACK,
        FRONTANDBACK
    }

    public struct layer_description
    {
        public LAYERTYPE layer_type;
        public int start_neurons_num;
        public int adding_neurons_num;
        public float dropout_perc;
        public float neurons_adding_timewait;

        public layer_description(LAYERTYPE Layer_type, int Start_neurons_num,
            int Adding_neurons_num = 0, int Dropout_perc = 0, float Neurons_adding_timewait = 1f)
        {
            layer_type = Layer_type;
            start_neurons_num = Start_neurons_num;
            adding_neurons_num = Adding_neurons_num;
            dropout_perc = Dropout_perc;
            neurons_adding_timewait = Neurons_adding_timewait;
        }
    }


    public struct color_prior
    {
        public Color color;
        public int prior;
        public bool continue_proc;
        public float rate;
        public color_prior(Color c, int p, bool cont, float r)
        {
            color = c;
            prior = p;
            continue_proc = cont;
            rate = r;
        }
    }
}


