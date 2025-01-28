#include "gelu.hpp"

#include <hls_math.h>
#include <hls_vector.h>

typedef ap_ufixed<fm_t::width - fm_t::iwidth, 0> fm_frac_t;
static const fm_frac_t GELU_DELTA_TABLE[] =
		{ 0.0, 0.015235424041748047, 0.029692649841308594, 0.043373823165893555,
				0.056282758712768555, 0.06842470169067383, 0.07980632781982422,
				0.09043622016906738, 0.10032343864440918, 0.10947918891906738,
				0.11791563034057617, 0.12564659118652344, 0.13268637657165527,
				0.13905096054077148, 0.14475750923156738, 0.14982390403747559,
				0.15426874160766602, 0.1581120491027832, 0.16137433052062988,
				0.164076566696167, 0.16624093055725098, 0.16788959503173828,
				0.16904520988464355, 0.16973090171813965, 0.16997051239013672,
				0.16978740692138672, 0.1692051887512207, 0.16824769973754883,
				0.16693854331970215, 0.16530156135559082, 0.1633601188659668,
				0.16113710403442383, 0.15865516662597656, 0.15593719482421875,
				0.15300464630126953, 0.14987921714782715, 0.14658141136169434,
				0.14313149452209473, 0.13954925537109375, 0.13585352897644043,
				0.13206219673156738, 0.12819290161132812, 0.12426185607910156,
				0.1202852725982666, 0.11627793312072754, 0.11225390434265137,
				0.10822653770446777, 0.10420823097229004, 0.10021090507507324,
				0.09624481201171875, 0.09232044219970703, 0.08844685554504395,
				0.08463215827941895, 0.08088397979736328, 0.0772092342376709,
				0.07361388206481934, 0.07010340690612793, 0.06668257713317871,
				0.0633549690246582, 0.06012439727783203, 0.05699324607849121,
				0.05396389961242676, 0.05103778839111328, 0.04821658134460449,
				0.04550027847290039, 0.04288959503173828, 0.04038381576538086,
				0.037982940673828125, 0.03568577766418457, 0.03349113464355469,
				0.031397342681884766, 0.029402494430541992,
				0.027505159378051758, 0.025702476501464844,
				0.023992300033569336, 0.02237224578857422, 0.020839452743530273,
				0.01939105987548828, 0.018024444580078125, 0.016736268997192383,
				0.015524148941040039, 0.014384746551513672,
				0.013314962387084961, 0.012311935424804688,
				0.011372566223144531, 0.010494232177734375,
				0.009673595428466797, 0.008907794952392578,
				0.008194446563720703, 0.00753021240234375, 0.006912946701049805,
				0.006339550018310547, 0.0058078765869140625,
				0.005315303802490234, 0.0048596858978271484,
				0.0044384002685546875, 0.004049777984619141,
				0.0036911964416503906, 0.0033609867095947266,
				0.003057241439819336, 0.002778291702270508,
				0.002521991729736328, 0.002287149429321289,
				0.0020720958709716797, 0.0018754005432128906,
				0.0016956329345703125, 0.001531362533569336,
				0.0013818740844726562, 0.001245737075805664,
				0.0011217594146728516, 0.0010089874267578125,
				0.0009069442749023438, 0.0008141994476318359,
				0.0007302761077880859, 0.0006542205810546875,
				0.0005857944488525391, 0.0005238056182861328,
				0.0004677772521972656, 0.00041747093200683594,
				0.00037217140197753906, 0.00033164024353027344,
				0.0002949237823486328, 0.00026226043701171875,
				0.00023293495178222656, 0.00020647048950195312,
				0.00018310546875, 0.0001621246337890625, 0.00014328956604003906,
				0.0001266002655029297, 0.00011181831359863281,
				9.870529174804688e-05, 8.678436279296875e-05,
				7.653236389160156e-05, 6.723403930664062e-05,
				5.91278076171875e-05, 5.173683166503906e-05,
				4.553794860839844e-05, 3.981590270996094e-05,
				3.4809112548828125e-05, 3.0517578125e-05,
				2.6464462280273438e-05, 2.3126602172851562e-05,
				2.0265579223632812e-05, 1.7642974853515625e-05,
				1.52587890625e-05, 1.33514404296875e-05, 1.1444091796875e-05,
				1.0013580322265625e-05, 8.58306884765625e-06,
				7.3909759521484375e-06, 6.4373016357421875e-06,
				5.4836273193359375e-06, 4.76837158203125e-06,
				4.0531158447265625e-06, 3.5762786865234375e-06,
				3.0994415283203125e-06, 2.6226043701171875e-06,
				2.384185791015625e-06, 1.9073486328125e-06,
				1.6689300537109375e-06, 1.430511474609375e-06,
				1.1920928955078125e-06, 9.5367431640625e-07,
				9.5367431640625e-07, 7.152557373046875e-07,
				7.152557373046875e-07, 4.76837158203125e-07,
				4.76837158203125e-07, 4.76837158203125e-07,
				2.384185791015625e-07, 2.384185791015625e-07,
				2.384185791015625e-07, 2.384185791015625e-07,
				2.384185791015625e-07, 2.384185791015625e-07,
				2.384185791015625e-07, 0.0 };

static constexpr unsigned int GELU_DELTA_TABLE_SIZE = sizeof(GELU_DELTA_TABLE)
		/ sizeof(GELU_DELTA_TABLE[0]);
typedef ap_uint<bitcount(GELU_DELTA_TABLE_SIZE - 1)> gelu_delta_table_index_t;

static const fm_t GELU_DELTA_TABLE_STEP = 0.03125;
static const fm_t GELU_DELTA_TABLE_MAX = GELU_DELTA_TABLE_STEP
		* (GELU_DELTA_TABLE_SIZE - 1);

fm_t gelu(fm_t x) {
#pragma HLS inline off
#pragma HLS bind_storage variable=GELU_DELTA_TABLE type=ROM_NP

	fm_t relu = ap_fixed_relu(x);
	fm_t x_abs = hls::signbit(x) ? fm_t(-x) : x;

	if (x_abs >= GELU_DELTA_TABLE_MAX)
		return relu;

	auto index_exact = x_abs / GELU_DELTA_TABLE_STEP;
	gelu_delta_table_index_t index = index_exact;

	fm_frac_t a = GELU_DELTA_TABLE[index];
	fm_frac_t b = GELU_DELTA_TABLE[index + 1];
	fm_frac_t t = index_exact - index;

	fm_frac_t gelu_delta = a + t * (b - a);
	return relu - gelu_delta;
}

template<size_t N>
hls::vector<fm_t, N> gelu(hls::vector<fm_t, N> x) {
	hls::vector<fm_t, N> y;
	gelu_loop: for (size_t i = 0; i < N; i++) {
#pragma HLS unroll
		y[i] = gelu(x[i]);
	}
	return y;
}

template hls::vector<fm_t, LINEAR_OUT_SIZE> gelu(
		hls::vector<fm_t, LINEAR_OUT_SIZE> x);
